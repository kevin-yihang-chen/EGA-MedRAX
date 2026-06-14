"""EvidenceGraph builder and torch-geometric exporter.

The graph itself lives inside ``schemas.EvidenceGraph``. This module gives
us two extra things:

  1. ``GraphBuilder`` - a convenience facade that wires a Collector + a
     rule pre-pass + a feature-tensor builder into a single call. The
     agent loop talks to this object, not to the lower-level pieces.
  2. ``build_torch_data`` - converts an EvidenceGraph to a PyG
     ``HeteroData`` for the learned verifier. Kept here so the schema
     module stays torch-free.

The feature design is:
  Claim node feature: [type one-hot (8), polarity one-hot (3), num_evidence,
                       num_supporting, num_contradicting]
  Evidence node feature: [evidence_type one-hot (7), proposition one-hot (4),
                          score, calibrated_score, uncertainty, has_region]
  Region node feature: [side one-hot (3), zone one-hot (3), bbox(4)]
  Edge types: (evidence, supports, claim),
              (evidence, contradicts, claim),
              (evidence, grounds, region),
              (claim, relates, claim)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import (
    ClaimType,
    Evidence,
    EvidenceGraph,
    EvidenceType,
    Polarity,
    Proposition,
    Region,
)
from .rules import MedicalConsistencyRules, RuleViolation


CLAIM_TYPE_LIST: List[str] = [c.value for c in ClaimType]
POLARITY_LIST: List[str] = [p.value for p in Polarity]
EVIDENCE_TYPE_LIST: List[str] = [e.value for e in EvidenceType]
PROPOSITION_LIST: List[str] = [p.value for p in Proposition]
SIDE_LIST: List[str] = ["left", "right", "midline"]
ZONE_LIST: List[str] = ["upper", "mid", "lower"]


# ---------------------------------------------------------------------------
# Builder facade
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Aggregates the post-collection steps.

    The agent code calls ``finalize`` after the collector has populated
    the graph. ``finalize`` runs the rule pass, attaches the resulting
    violations as a graph attribute, and returns the graph plus the
    violation list. The violations are then consumed by both the rule
    verifier and the abstention policy.
    """

    def __init__(self, rules: Optional[MedicalConsistencyRules] = None):
        self.rules = rules or MedicalConsistencyRules()

    def finalize(self, graph: EvidenceGraph) -> Tuple[EvidenceGraph, List[RuleViolation]]:
        violations = self.rules(graph)
        return graph, violations


# ---------------------------------------------------------------------------
# Vector / torch_geometric export
# ---------------------------------------------------------------------------


def _one_hot(value: Any, vocab: List[str]) -> List[float]:
    vec = [0.0] * len(vocab)
    try:
        idx = vocab.index(value)
        vec[idx] = 1.0
    except ValueError:
        pass
    return vec


def claim_features(graph: EvidenceGraph, claim_id: str) -> List[float]:
    claim = graph.claims[claim_id]
    evidences = graph.evidence_for(claim_id)
    supporting = [e for e in evidences if e.proposition == Proposition.SUPPORTS]
    contradicting = [e for e in evidences if e.proposition == Proposition.CONTRADICTS]
    return (
        _one_hot(claim.type.value, CLAIM_TYPE_LIST)
        + _one_hot(claim.polarity.value, POLARITY_LIST)
        + [
            float(len(evidences)),
            float(len(supporting)),
            float(len(contradicting)),
            float(sum(e.calibrated_score for e in supporting)),
            float(sum(e.calibrated_score for e in contradicting)),
        ]
    )


def evidence_features(ev: Evidence) -> List[float]:
    return (
        _one_hot(ev.evidence_type.value, EVIDENCE_TYPE_LIST)
        + _one_hot(ev.proposition.value, PROPOSITION_LIST)
        + [
            float(ev.score),
            float(ev.calibrated_score),
            float(ev.uncertainty),
            1.0 if ev.region_id else 0.0,
        ]
    )


def region_features(region: Region) -> List[float]:
    bbox = list(region.bbox) if region.bbox else [0.0, 0.0, 0.0, 0.0]
    return (
        _one_hot(region.side or "midline", SIDE_LIST)
        + _one_hot(region.zone or "mid", ZONE_LIST)
        + [float(x) for x in bbox]
    )


def build_torch_data(graph: EvidenceGraph) -> Any:
    """Materialize an EvidenceGraph as a torch_geometric HeteroData.

    Returns ``None`` if torch_geometric is unavailable. The verifier can
    fall back to its MLP variant in that case, so the graph module never
    hard-requires torch_geometric to be installed.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        return None

    data = HeteroData()

    claim_ids = list(graph.claims.keys())
    evidence_ids = list(graph.evidence.keys())
    region_ids = list(graph.regions.keys())
    claim_idx = {cid: i for i, cid in enumerate(claim_ids)}
    evidence_idx = {eid: i for i, eid in enumerate(evidence_ids)}
    region_idx = {rid: i for i, rid in enumerate(region_ids)}

    data["claim"].x = torch.tensor(
        [claim_features(graph, cid) for cid in claim_ids], dtype=torch.float
    ) if claim_ids else torch.empty((0, len(CLAIM_TYPE_LIST) + len(POLARITY_LIST) + 5))
    data["evidence"].x = torch.tensor(
        [evidence_features(graph.evidence[eid]) for eid in evidence_ids], dtype=torch.float
    ) if evidence_ids else torch.empty((0, len(EVIDENCE_TYPE_LIST) + len(PROPOSITION_LIST) + 4))
    data["region"].x = torch.tensor(
        [region_features(graph.regions[rid]) for rid in region_ids], dtype=torch.float
    ) if region_ids else torch.empty((0, len(SIDE_LIST) + len(ZONE_LIST) + 4))

    support_src, support_dst = [], []
    contradict_src, contradict_dst = [], []
    grounds_src, grounds_dst = [], []
    for eid, ev in graph.evidence.items():
        if ev.claim_id not in claim_idx:
            continue
        ei = evidence_idx[eid]
        ci = claim_idx[ev.claim_id]
        if ev.proposition == Proposition.SUPPORTS:
            support_src.append(ei); support_dst.append(ci)
        elif ev.proposition == Proposition.CONTRADICTS:
            contradict_src.append(ei); contradict_dst.append(ci)
        if ev.region_id in region_idx:
            grounds_src.append(ei); grounds_dst.append(region_idx[ev.region_id])

    relates_src, relates_dst = [], []
    for parent, child, _ in graph.relates:
        if parent in claim_idx and child in claim_idx:
            relates_src.append(claim_idx[parent]); relates_dst.append(claim_idx[child])

    data["evidence", "supports", "claim"].edge_index = torch.tensor(
        [support_src, support_dst], dtype=torch.long
    ) if support_src else torch.empty((2, 0), dtype=torch.long)
    data["evidence", "contradicts", "claim"].edge_index = torch.tensor(
        [contradict_src, contradict_dst], dtype=torch.long
    ) if contradict_src else torch.empty((2, 0), dtype=torch.long)
    data["evidence", "grounds", "region"].edge_index = torch.tensor(
        [grounds_src, grounds_dst], dtype=torch.long
    ) if grounds_src else torch.empty((2, 0), dtype=torch.long)
    data["claim", "relates", "claim"].edge_index = torch.tensor(
        [relates_src, relates_dst], dtype=torch.long
    ) if relates_src else torch.empty((2, 0), dtype=torch.long)

    data["claim"].ids = claim_ids
    data["evidence"].ids = evidence_ids
    data["region"].ids = region_ids
    return data
