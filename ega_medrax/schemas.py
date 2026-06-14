"""Core data structures for EGA-MedRAX.

Everything the agent reasons over is expressed with these objects:

  * Claim          - an atomic medical proposition to be judged
  * Region         - an anatomical / visual anchor (bbox, mask, side tag)
  * Evidence      - a normalized observation produced by one tool for one claim
  * EvidenceGraph - the claim/evidence/region graph with typed edges
  * VerifierOutput - per-claim truth, uncertainty, conflict, sufficiency
  * AbstentionDecision - the final answer/abstain choice with a structured reason

These objects are deliberately framework-free (no torch / no langchain) so they
can be used by the rule verifier, the learned verifier, and the training
pipeline without circular imports.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """Medical claim taxonomy.

    The taxonomy is small on purpose: it covers what we need to wire tools
    to claims (a classifier is great for presence, a segmentation is great
    for extent, grounding is great for localization, etc.).
    """

    FINDING = "finding"            # presence/absence of a pathology
    LOCATION = "location"          # laterality / anatomical site
    SEVERITY = "severity"          # mild / moderate / severe
    EXTENT = "extent"              # area / measurement
    COMPARISON = "comparison"      # worsened / improved vs prior
    DIAGNOSIS = "diagnosis"        # high-level diagnostic conclusion
    DEVICE = "device"              # support device / tube position
    METADATA = "metadata"          # technical / DICOM attribute


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


@dataclass
class Claim:
    """An atomic, judge-able medical proposition.

    A Claim is the unit of truth in EGA-MedRAX. The verifier outputs a
    truth / uncertainty / sufficiency triple *per claim*, and the final
    answer is a function of the claim graph - not of free-form tool text.
    """

    text: str                                       # human-readable statement
    type: ClaimType = ClaimType.FINDING
    subject: str = ""                               # e.g. "pleural effusion"
    attribute: str = ""                             # e.g. "present", "mild"
    anatomy: str = ""                               # e.g. "left lung base"
    time: str = "current"                           # "current" / "prior" / "delta"
    polarity: Polarity = Polarity.POSITIVE
    parents: List[str] = field(default_factory=list)  # ids of logically prior claims
    id: str = field(default_factory=lambda: f"c_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type.value,
            "subject": self.subject,
            "attribute": self.attribute,
            "anatomy": self.anatomy,
            "time": self.time,
            "polarity": self.polarity.value,
            "parents": list(self.parents),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            id=data.get("id", f"c_{uuid.uuid4().hex[:8]}"),
            text=data["text"],
            type=ClaimType(data.get("type", "finding")),
            subject=data.get("subject", ""),
            attribute=data.get("attribute", ""),
            anatomy=data.get("anatomy", ""),
            time=data.get("time", "current"),
            polarity=Polarity(data.get("polarity", "positive")),
            parents=list(data.get("parents", [])),
        )


# ---------------------------------------------------------------------------
# Regions (visual anchors)
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """A visual / anatomical anchor.

    Regions deliberately mix three kinds of localization because real tools
    return all three: a bounding box, a mask reference, and a coarse
    anatomical tag (e.g. "left lower zone"). The verifier uses laterality
    and zone tags for rule checks; the learned verifier can use the box.
    """

    label: str                                     # e.g. "left base", "cardiac silhouette"
    side: Optional[str] = None                     # "left" / "right" / "bilateral"
    zone: Optional[str] = None                     # "upper" / "mid" / "lower" / ...
    bbox: Optional[Tuple[float, float, float, float]] = None   # x1,y1,x2,y2 normalized
    mask_ref: Optional[str] = None                 # path / id of a stored mask
    id: str = field(default_factory=lambda: f"r_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "side": self.side,
            "zone": self.zone,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "mask_ref": self.mask_ref,
        }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):
    CLASSIFICATION = "cls"
    SEGMENTATION = "seg"
    GROUNDING = "grounding"
    VQA = "vqa"
    REPORT = "report"
    METADATA = "metadata"
    RETRIEVED_TEXT = "retrieved_text"


class Proposition(str, Enum):
    """What an evidence node says about its claim.

    Critically we allow CONTRADICTS and INSUFFICIENT - the most dangerous
    failure mode in medical agents is silently dropping conflicting tool
    output, so the graph models conflict as a first-class edge type.
    """

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Evidence:
    """A normalized observation produced by exactly one tool for one claim.

    Two key design choices follow the EGA design doc:

      * `proposition` is structured (supports / contradicts / insufficient /
        not_applicable), not free text - so the verifier can sum support and
        contradiction scores explicitly.
      * `calibrated_score` is separate from `score` so a tool-reliability
        prior can rescale raw confidence per (tool, claim_type) without
        losing the original value.
    """

    claim_id: str
    tool_name: str
    evidence_type: EvidenceType
    proposition: Proposition
    score: float                                  # raw tool confidence
    calibrated_score: float                       # after tool-reliability prior
    uncertainty: float = 0.0                      # epistemic uncertainty
    region_id: Optional[str] = None               # link to a Region node
    text: str = ""                                # short rationale
    provenance: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "tool_name": self.tool_name,
            "evidence_type": self.evidence_type.value,
            "proposition": self.proposition.value,
            "score": self.score,
            "calibrated_score": self.calibrated_score,
            "uncertainty": self.uncertainty,
            "region_id": self.region_id,
            "text": self.text,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Evidence graph
# ---------------------------------------------------------------------------


@dataclass
class EvidenceGraph:
    """The reasoning state of the agent.

    Three node tables + four edge types. Stored as plain dicts so the graph
    is JSON-serializable for logging and evaluation. The learned verifier
    converts this into a torch_geometric HeteroData on demand.

    Edge types:
      * supports(e -> c)
      * contradicts(e -> c)
      * grounds(e -> r)
      * relates(c_i -> c_j)         logical relation between claims
    """

    claims: Dict[str, Claim] = field(default_factory=dict)
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    regions: Dict[str, Region] = field(default_factory=dict)
    relates: List[Tuple[str, str, str]] = field(default_factory=list)
    # (parent_claim_id, child_claim_id, relation: "implies" | "compares" | "depends")

    # --- mutation helpers ------------------------------------------------

    def add_claim(self, claim: Claim) -> None:
        self.claims[claim.id] = claim

    def add_region(self, region: Region) -> None:
        self.regions[region.id] = region

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence[ev.id] = ev

    def add_relation(self, parent: str, child: str, relation: str = "implies") -> None:
        self.relates.append((parent, child, relation))

    # --- accessors -------------------------------------------------------

    def evidence_for(self, claim_id: str) -> List[Evidence]:
        return [e for e in self.evidence.values() if e.claim_id == claim_id]

    def supporting(self, claim_id: str) -> List[Evidence]:
        return [e for e in self.evidence_for(claim_id)
                if e.proposition == Proposition.SUPPORTS]

    def contradicting(self, claim_id: str) -> List[Evidence]:
        return [e for e in self.evidence_for(claim_id)
                if e.proposition == Proposition.CONTRADICTS]

    def region_of(self, evidence_id: str) -> Optional[Region]:
        ev = self.evidence.get(evidence_id)
        if ev is None or ev.region_id is None:
            return None
        return self.regions.get(ev.region_id)

    # --- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": {cid: c.to_dict() for cid, c in self.claims.items()},
            "evidence": {eid: e.to_dict() for eid, e in self.evidence.items()},
            "regions": {rid: r.to_dict() for rid, r in self.regions.items()},
            "relates": [list(t) for t in self.relates],
        }


# ---------------------------------------------------------------------------
# Verifier output and abstention
# ---------------------------------------------------------------------------


@dataclass
class VerifierOutput:
    """Per-claim verifier head output.

    `sufficiency` is the verifier's own answer to "did we collect enough
    evidence here", which is what drives adaptive tool acquisition.
    """

    claim_id: str
    p_true: float                                 # P(claim is true)
    p_false: float                                # P(claim is false)
    uncertainty: float                            # epistemic uncertainty
    conflict: float                               # 0..1, evidence disagreement
    sufficiency: float                            # 0..1, do we have enough evidence
    rationale: str = ""                           # short human-readable note

    @property
    def confidence(self) -> float:
        """Margin between true/false probability, in [0, 1]."""
        return abs(self.p_true - self.p_false)


class AbstentionReason(str, Enum):
    NONE = "none"
    INSUFFICIENT_VISUAL_EVIDENCE = "insufficient_visual_evidence"
    CROSS_TOOL_CONFLICT_UNRESOLVED = "cross_tool_conflict_unresolved"
    MISSING_REQUIRED_CONTEXT = "missing_required_context"
    LOW_CONFIDENCE = "low_confidence"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


@dataclass
class AbstentionDecision:
    """Final answer-or-abstain decision.

    Captures both the binary decision and the structured reason, so the
    agent's output can be audited: an abstention with reason
    `CROSS_TOOL_CONFLICT_UNRESOLVED` is a very different signal to a
    clinician than `INSUFFICIENT_VISUAL_EVIDENCE`.
    """

    abstain: bool
    reason: AbstentionReason
    answer: str = ""
    confidence: float = 0.0
    abstention_score: float = 0.0                 # raw value before threshold
    breakdown: Dict[str, float] = field(default_factory=dict)
    supporting_claim_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstain": self.abstain,
            "reason": self.reason.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "abstention_score": self.abstention_score,
            "breakdown": self.breakdown,
            "supporting_claim_ids": list(self.supporting_claim_ids),
        }
