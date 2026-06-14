"""Structured abstention policy.

Computes the abstention decision from per-claim verifier outputs plus
graph-level diagnostics. Crucially the decision is *structured* - we
report not just "abstain yes/no" but a typed reason that maps onto the
three taxonomic abstention classes in the design doc:

  * INSUFFICIENT_VISUAL_EVIDENCE   - low coverage / low sufficiency
  * CROSS_TOOL_CONFLICT_UNRESOLVED - high conflict among supporting tools
  * MISSING_REQUIRED_CONTEXT       - e.g. COMPARISON without prior

The actual scalar abstention score combines five signals:
    A = w_c * (1 - max_confidence)
      + w_o * mean_conflict
      + w_v * (1 - mean_sufficiency)
      + w_d * (1 - tool_agreement)
      + w_ood * ood_score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    AbstentionDecision,
    AbstentionReason,
    Claim,
    ClaimType,
    EvidenceGraph,
    Proposition,
    VerifierOutput,
)
from ..graph.rules import RuleViolation


logger = logging.getLogger(__name__)


@dataclass
class AbstentionConfig:
    confidence_w: float = 0.30
    conflict_w: float = 0.25
    sufficiency_w: float = 0.25
    agreement_w: float = 0.10
    ood_w: float = 0.10
    threshold: float = 0.55
    min_supporting_tools: int = 2


class AbstentionPolicy:
    """The decision head."""

    def __init__(self, config: Optional[AbstentionConfig] = None):
        self.config = config or AbstentionConfig()

    def decide(
        self,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
        violations: Optional[List[RuleViolation]] = None,
        ood_score: float = 0.0,
        question: str = "",
    ) -> AbstentionDecision:
        if not verifier_outputs:
            return AbstentionDecision(
                abstain=True,
                reason=AbstentionReason.INSUFFICIENT_VISUAL_EVIDENCE,
                answer="",
                confidence=0.0,
                abstention_score=1.0,
                breakdown={"reason": "no claims to verify"},
            )

        primary = self._primary_claim(graph, verifier_outputs)
        primary_id = primary.claim_id if primary else next(iter(verifier_outputs))

        max_conf = max(v.confidence for v in verifier_outputs.values())
        mean_conflict = _mean(v.conflict for v in verifier_outputs.values())
        mean_sufficiency = _mean(v.sufficiency for v in verifier_outputs.values())
        tool_agreement = self._tool_agreement(graph)

        score = (
            self.config.confidence_w * (1.0 - max_conf)
            + self.config.conflict_w * mean_conflict
            + self.config.sufficiency_w * (1.0 - mean_sufficiency)
            + self.config.agreement_w * (1.0 - tool_agreement)
            + self.config.ood_w * ood_score
        )

        reason = self._infer_reason(
            graph, verifier_outputs, violations or [],
            mean_conflict=mean_conflict, mean_sufficiency=mean_sufficiency,
            tool_agreement=tool_agreement, ood_score=ood_score,
        )
        abstain = score >= self.config.threshold or reason != AbstentionReason.NONE

        breakdown = {
            "max_confidence": float(max_conf),
            "mean_conflict": float(mean_conflict),
            "mean_sufficiency": float(mean_sufficiency),
            "tool_agreement": float(tool_agreement),
            "ood_score": float(ood_score),
            "threshold": float(self.config.threshold),
        }

        return AbstentionDecision(
            abstain=abstain,
            reason=reason if abstain else AbstentionReason.NONE,
            answer="",  # filled by the AnswerHead in agent.py
            confidence=float(max_conf),
            abstention_score=float(score),
            breakdown=breakdown,
            supporting_claim_ids=[primary_id],
        )

    # ------------------------------------------------------------------
    # Reason inference - structured, not just thresholded
    # ------------------------------------------------------------------

    def _infer_reason(
        self,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
        violations: List[RuleViolation],
        *,
        mean_conflict: float,
        mean_sufficiency: float,
        tool_agreement: float,
        ood_score: float,
    ) -> AbstentionReason:
        # Order matters: more specific reasons win over generic low-confidence.
        if any(v.rule == "comparison_needs_prior" for v in violations):
            return AbstentionReason.MISSING_REQUIRED_CONTEXT
        if mean_conflict > 0.5 or tool_agreement < 0.4:
            return AbstentionReason.CROSS_TOOL_CONFLICT_UNRESOLVED
        if mean_sufficiency < 0.45 or len(graph.evidence) < self.config.min_supporting_tools:
            return AbstentionReason.INSUFFICIENT_VISUAL_EVIDENCE
        if ood_score > 0.7:
            return AbstentionReason.OUT_OF_DISTRIBUTION
        max_conf = max(v.confidence for v in verifier_outputs.values())
        if max_conf < 0.3:
            return AbstentionReason.LOW_CONFIDENCE
        return AbstentionReason.NONE

    # ------------------------------------------------------------------
    # Sub-signals
    # ------------------------------------------------------------------

    def _primary_claim(
        self,
        graph: EvidenceGraph,
        verifier_outputs: Dict[str, VerifierOutput],
    ) -> Optional[VerifierOutput]:
        """Pick the most question-defining claim to attach to the answer."""
        # Prefer top-level findings or diagnoses with the highest confidence.
        def _priority(c: Claim) -> int:
            return {
                ClaimType.DIAGNOSIS: 4, ClaimType.FINDING: 3,
                ClaimType.COMPARISON: 2, ClaimType.LOCATION: 1,
            }.get(c.type, 0)
        ranked: List[Tuple[int, float, VerifierOutput]] = []
        for cid, v in verifier_outputs.items():
            claim = graph.claims.get(cid)
            if claim is None:
                continue
            ranked.append((_priority(claim), v.confidence, v))
        if not ranked:
            return None
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked[0][2]

    def _tool_agreement(self, graph: EvidenceGraph) -> float:
        """Average per-claim agreement.

        For each claim with >= 2 distinct tools, agreement is
        1 - (# minority proposition / # evidences). Claims with one tool
        contribute the neutral 0.5.
        """
        if not graph.claims:
            return 0.5
        scores: List[float] = []
        for cid in graph.claims:
            evs = graph.evidence_for(cid)
            if not evs:
                continue
            by_tool: Dict[str, Proposition] = {}
            for ev in evs:
                if ev.proposition in (Proposition.SUPPORTS, Proposition.CONTRADICTS):
                    by_tool[ev.tool_name] = ev.proposition
            if len(by_tool) < 2:
                scores.append(0.5)
                continue
            supports = sum(1 for p in by_tool.values() if p == Proposition.SUPPORTS)
            contradicts = len(by_tool) - supports
            agree = max(supports, contradicts) / len(by_tool)
            scores.append(float(agree))
        return _mean(scores) if scores else 0.5


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
