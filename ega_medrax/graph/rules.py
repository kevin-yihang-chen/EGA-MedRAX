"""Medical consistency rules.

These rules implement the rule-guided pathway from Section "Rule-guided
graph inference" of the design doc. They run as a fast pre-pass that:

  * marks an Evidence/Claim pair as a rule-violation (e.g. laterality
    mismatch between a left-sided claim and a right-sided segmentation),
    pushing the verifier toward CONTRADICTS;
  * raises the conflict score on COMPARISON claims that lack any prior-
    study evidence;
  * caps confidence on DIAGNOSIS claims whose finding parents are all
    unverified.

We expose the violations as a list of typed events so the abstention
policy can show them as a structured "reason" - this is the auditability
property that makes the agent paper-worthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..schemas import (
    Claim,
    ClaimType,
    EvidenceGraph,
    Polarity,
    Proposition,
)


@dataclass(frozen=True)
class RuleViolation:
    rule: str
    claim_id: str
    evidence_id: str = ""
    severity: float = 1.0
    detail: str = ""


class MedicalConsistencyRules:
    """Collection of rule predicates over the evidence graph.

    Each rule returns zero or more RuleViolations. Rules are intentionally
    small and orthogonal so they're easy to extend per institution / per
    benchmark.
    """

    def __init__(self) -> None:
        self._rules = [
            self._rule_laterality_mismatch,
            self._rule_comparison_needs_prior,
            self._rule_diagnosis_needs_finding,
            self._rule_low_evidence_high_polarity,
        ]

    def __call__(self, graph: EvidenceGraph) -> List[RuleViolation]:
        return self.evaluate(graph)

    def evaluate(self, graph: EvidenceGraph) -> List[RuleViolation]:
        violations: List[RuleViolation] = []
        for rule in self._rules:
            violations.extend(rule(graph))
        return violations

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _rule_laterality_mismatch(self, graph: EvidenceGraph) -> List[RuleViolation]:
        """A left-sided claim cannot be supported by right-sided evidence."""
        out: List[RuleViolation] = []
        for claim in graph.claims.values():
            target_side = _claim_side(claim)
            if not target_side:
                continue
            for ev in graph.evidence_for(claim.id):
                if ev.proposition != Proposition.SUPPORTS:
                    continue
                region = graph.region_of(ev.id) if ev.region_id else None
                if region is None or region.side is None or region.side in ("midline", "bilateral"):
                    continue
                if region.side != target_side:
                    out.append(RuleViolation(
                        rule="laterality_mismatch",
                        claim_id=claim.id,
                        evidence_id=ev.id,
                        severity=1.0,
                        detail=f"claim side={target_side} vs evidence side={region.side}",
                    ))
        return out

    def _rule_comparison_needs_prior(self, graph: EvidenceGraph) -> List[RuleViolation]:
        """COMPARISON claims need at least one piece of prior-study evidence."""
        out: List[RuleViolation] = []
        for claim in graph.claims.values():
            if claim.type != ClaimType.COMPARISON:
                continue
            has_prior = any(
                ev.provenance.get("time") == "prior"
                or "prior" in str(ev.text).lower()
                or "previous" in str(ev.text).lower()
                for ev in graph.evidence_for(claim.id)
            )
            if not has_prior:
                out.append(RuleViolation(
                    rule="comparison_needs_prior",
                    claim_id=claim.id,
                    severity=0.8,
                    detail="no prior-study evidence linked",
                ))
        return out

    def _rule_diagnosis_needs_finding(self, graph: EvidenceGraph) -> List[RuleViolation]:
        """A DIAGNOSIS claim's finding parents must each have supporting evidence."""
        out: List[RuleViolation] = []
        for claim in graph.claims.values():
            if claim.type != ClaimType.DIAGNOSIS or not claim.parents:
                continue
            for pid in claim.parents:
                parent = graph.claims.get(pid)
                if parent is None or parent.type != ClaimType.FINDING:
                    continue
                if not graph.supporting(pid):
                    out.append(RuleViolation(
                        rule="diagnosis_without_finding",
                        claim_id=claim.id,
                        severity=0.6,
                        detail=f"finding parent {pid} has no supporting evidence",
                    ))
        return out

    def _rule_low_evidence_high_polarity(self, graph: EvidenceGraph) -> List[RuleViolation]:
        """Negative-polarity claims with zero contradicting evidence cannot be confidently negative."""
        out: List[RuleViolation] = []
        for claim in graph.claims.values():
            if claim.polarity != Polarity.NEGATIVE:
                continue
            if not graph.supporting(claim.id):
                out.append(RuleViolation(
                    rule="bare_negative_claim",
                    claim_id=claim.id,
                    severity=0.5,
                    detail="negative claim has no evidence ruling the finding out",
                ))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim_side(claim: Claim) -> str:
    anat = (claim.anatomy or "").lower()
    if "left" in anat:
        return "left"
    if "right" in anat:
        return "right"
    if "bilateral" in anat or "both" in anat:
        return "bilateral"
    attr = (claim.attribute or "").lower()
    if attr in ("left", "right", "bilateral"):
        return attr
    return ""


def violations_to_dict(violations: List[RuleViolation]) -> Dict[str, List[Dict[str, object]]]:
    """Group violations by claim id for compact logging."""
    out: Dict[str, List[Dict[str, object]]] = {}
    for v in violations:
        out.setdefault(v.claim_id, []).append({
            "rule": v.rule,
            "evidence_id": v.evidence_id,
            "severity": v.severity,
            "detail": v.detail,
        })
    return out
