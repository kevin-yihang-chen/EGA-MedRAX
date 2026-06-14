"""Dataset wrapper for the pseudo-labelled EGA graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ega_medrax.graph.builder import (
    CLAIM_TYPE_LIST,
    POLARITY_LIST,
    build_torch_data,
    claim_features,
)
from ega_medrax.schemas import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceGraph,
    EvidenceType,
    Polarity,
    Proposition,
    Region,
)


class EGAGraphDataset(Dataset):
    """Loads the JSON pseudo-examples produced by ``build_pseudo_dataset``.

    Returns
    -------
    sample : dict
        keys:
          - 'graph_data'      : torch_geometric HeteroData (if available) or None
          - 'flat_features'   : tensor [n_claims, feat_dim]
          - 'claim_labels'    : tensor [n_claims] (long, -1 for unknown)
          - 'conflict_labels' : tensor [n_claims] (float)
          - 'claim_ids'       : list[str]
    """

    def __init__(self, root: str):
        self.paths = sorted(Path(root).glob("sample_*.json"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = json.loads(self.paths[idx].read_text())
        graph = _graph_from_dict(data["graph"])
        claim_ids = list(graph.claims.keys())

        flat = torch.tensor(
            [claim_features(graph, cid) for cid in claim_ids], dtype=torch.float,
        ) if claim_ids else torch.empty((0, len(CLAIM_TYPE_LIST) + len(POLARITY_LIST) + 5))

        claim_labels = torch.tensor(
            [int(data["claim_labels"].get(cid, -1)) for cid in claim_ids], dtype=torch.long,
        )
        conflict_labels = torch.tensor(
            [float(data["conflict_labels"].get(cid, 0)) for cid in claim_ids], dtype=torch.float,
        )

        return {
            "graph_data": build_torch_data(graph),
            "flat_features": flat,
            "claim_labels": claim_labels,
            "conflict_labels": conflict_labels,
            "claim_ids": claim_ids,
            "question": data.get("question", ""),
            "image_path": data.get("image_path", ""),
            "answerable": bool(data.get("answerable", True)),
        }


# ---------------------------------------------------------------------------
# Dict -> EvidenceGraph
# ---------------------------------------------------------------------------


def _graph_from_dict(d: Dict[str, Any]) -> EvidenceGraph:
    g = EvidenceGraph()
    for cid, cdat in d.get("claims", {}).items():
        c = Claim.from_dict(cdat)
        g.claims[cid] = c
    for rid, rdat in d.get("regions", {}).items():
        g.regions[rid] = Region(
            id=rid, label=rdat["label"],
            side=rdat.get("side"), zone=rdat.get("zone"),
            bbox=tuple(rdat["bbox"]) if rdat.get("bbox") else None,
            mask_ref=rdat.get("mask_ref"),
        )
    for eid, edat in d.get("evidence", {}).items():
        g.evidence[eid] = Evidence(
            id=eid,
            claim_id=edat["claim_id"],
            tool_name=edat["tool_name"],
            evidence_type=EvidenceType(edat["evidence_type"]),
            proposition=Proposition(edat["proposition"]),
            score=float(edat["score"]),
            calibrated_score=float(edat["calibrated_score"]),
            uncertainty=float(edat.get("uncertainty", 0.0)),
            region_id=edat.get("region_id"),
            text=edat.get("text", ""),
            provenance=dict(edat.get("provenance", {})),
            timestamp=float(edat.get("timestamp", 0.0)),
        )
    for triple in d.get("relates", []):
        g.relates.append(tuple(triple))  # type: ignore[arg-type]
    return g
