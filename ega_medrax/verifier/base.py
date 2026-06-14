"""Verifier interfaces.

A verifier takes an EvidenceGraph (+ optional rule violations) and emits a
``VerifierOutput`` per claim. We define a tiny abstract base plus a
HybridVerifier that combines a rule pass with a learned pass - the
hybrid is the recommended production configuration from the design doc.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import EvidenceGraph, VerifierOutput
from ..graph.rules import RuleViolation


class BaseVerifier:
    """Common interface for rule, MLP, and graph-transformer verifiers."""

    def verify(
        self,
        graph: EvidenceGraph,
        violations: Optional[List[RuleViolation]] = None,
    ) -> Dict[str, VerifierOutput]:
        raise NotImplementedError


class HybridVerifier(BaseVerifier):
    """Combine a fast rule verifier with a learned verifier.

    Strategy
    --------
    1. Run the rule verifier to get a deterministic prior per claim.
    2. Run the learned verifier (if available) to get a soft estimate.
    3. Mix the two with a learnable scalar ``alpha`` (default 0.5). At
       inference time alpha defaults to 0.5; at train time the trainer
       updates only the learned component but can also fit alpha.

    A claim flagged by a hard rule (severity >= 1.0) gets its
    ``p_false`` clamped upward, mirroring the design-doc principle that
    laterality / temporal violations should not be softly overridden by
    a learned model.
    """

    def __init__(
        self,
        rule_verifier: BaseVerifier,
        learned_verifier: Optional[BaseVerifier] = None,
        alpha: float = 0.5,
        hard_rule_threshold: float = 1.0,
    ):
        self.rule_verifier = rule_verifier
        self.learned_verifier = learned_verifier
        self.alpha = alpha
        self.hard_rule_threshold = hard_rule_threshold

    def verify(
        self,
        graph: EvidenceGraph,
        violations: Optional[List[RuleViolation]] = None,
    ) -> Dict[str, VerifierOutput]:
        rule_out = self.rule_verifier.verify(graph, violations)
        if self.learned_verifier is None:
            return rule_out
        learned_out = self.learned_verifier.verify(graph, violations)
        merged: Dict[str, VerifierOutput] = {}
        for cid in graph.claims:
            r = rule_out.get(cid)
            l = learned_out.get(cid)
            if r is None and l is None:
                continue
            if r is None:
                merged[cid] = l  # type: ignore[assignment]
                continue
            if l is None:
                merged[cid] = r
                continue
            merged[cid] = self._mix(r, l, violations or [])
        return merged

    def _mix(
        self,
        rule_out: VerifierOutput,
        learned_out: VerifierOutput,
        violations: List[RuleViolation],
    ) -> VerifierOutput:
        a = self.alpha
        p_true = a * rule_out.p_true + (1 - a) * learned_out.p_true
        p_false = a * rule_out.p_false + (1 - a) * learned_out.p_false
        uncertainty = max(rule_out.uncertainty, learned_out.uncertainty)
        conflict = max(rule_out.conflict, learned_out.conflict)
        sufficiency = min(1.0, 0.5 * (rule_out.sufficiency + learned_out.sufficiency))

        hard_hit = any(
            v.claim_id == rule_out.claim_id and v.severity >= self.hard_rule_threshold
            for v in violations
        )
        if hard_hit:
            p_false = max(p_false, 0.7)
            p_true = min(p_true, 0.3)
            conflict = max(conflict, 0.6)

        return VerifierOutput(
            claim_id=rule_out.claim_id,
            p_true=float(p_true),
            p_false=float(p_false),
            uncertainty=float(uncertainty),
            conflict=float(conflict),
            sufficiency=float(sufficiency),
            rationale=(rule_out.rationale + " | " + learned_out.rationale).strip(" |"),
        )
