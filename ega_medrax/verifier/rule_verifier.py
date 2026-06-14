"""Rule-based claim verifier.

Translates evidence sums and rule violations into a per-claim
``VerifierOutput`` without any learned parameters. This is the deterministic
backbone used in:

  * version-1 (rule-augmented) deployments;
  * unit tests for the rest of the pipeline;
  * pseudo-labeling during data construction.

Math: for claim c with supporting score S and contradicting score C,

    p_true     = sigmoid( k * (S - C) ) gated by sufficiency
    p_false    = 1 - p_true             but raised by hard-rule hits
    conflict   = min(S, C) / max(S+C, eps)
    sufficiency = saturate(S + C)
    uncertainty = entropy(p_true) + (1 - sufficiency)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from ..schemas import EvidenceGraph, Proposition, VerifierOutput
from ..graph.rules import RuleViolation
from .base import BaseVerifier


class RuleVerifier(BaseVerifier):
    def __init__(self, gain: float = 4.0, eps: float = 1e-6):
        self.gain = gain
        self.eps = eps

    def verify(
        self,
        graph: EvidenceGraph,
        violations: Optional[List[RuleViolation]] = None,
    ) -> Dict[str, VerifierOutput]:
        violations = violations or []
        violations_by_claim: Dict[str, List[RuleViolation]] = defaultdict(list)
        for v in violations:
            violations_by_claim[v.claim_id].append(v)

        outputs: Dict[str, VerifierOutput] = {}
        for cid in graph.claims:
            outputs[cid] = self._verify_claim(graph, cid, violations_by_claim[cid])
        return outputs

    def _verify_claim(
        self,
        graph: EvidenceGraph,
        claim_id: str,
        claim_violations: List[RuleViolation],
    ) -> VerifierOutput:
        evidences = graph.evidence_for(claim_id)
        S = sum(e.calibrated_score for e in evidences if e.proposition == Proposition.SUPPORTS)
        C = sum(e.calibrated_score for e in evidences if e.proposition == Proposition.CONTRADICTS)

        # Hard rule violations on this claim push C upward.
        for v in claim_violations:
            if v.rule == "laterality_mismatch":
                C += 1.0 * v.severity
            else:
                C += 0.5 * v.severity

        margin = S - C
        p_true = _sigmoid(self.gain * margin)
        p_false = 1.0 - p_true
        conflict = (min(S, C) / max(S + C, self.eps))
        sufficiency = float(min(1.0, math.tanh(0.6 * (S + C))))
        uncertainty = float(_binary_entropy(p_true) + (1.0 - sufficiency))

        rationale = (
            f"S={S:.2f}, C={C:.2f}, conflict={conflict:.2f}, "
            f"sufficiency={sufficiency:.2f}, violations={len(claim_violations)}"
        )

        return VerifierOutput(
            claim_id=claim_id,
            p_true=float(p_true),
            p_false=float(p_false),
            uncertainty=float(min(uncertainty, 1.5)),
            conflict=float(conflict),
            sufficiency=sufficiency,
            rationale=rationale,
        )


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2)
