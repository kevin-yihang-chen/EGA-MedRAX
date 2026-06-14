"""Learned claim verifiers.

Two implementations:

  * MLPClaimVerifier - small per-claim MLP over a hand-pooled feature
    vector. Cheap, easy to train, depends only on torch.
  * GraphTransformerVerifier - a Hetero GNN / graph transformer over the
    EvidenceGraph, depends on torch_geometric. This is the version-2
    "main-conference" path from the design doc.

Both expose the BaseVerifier interface so the HybridVerifier can swap
between them without the agent caring.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..schemas import (
    Claim,
    ClaimType,
    EvidenceGraph,
    Polarity,
    Proposition,
    VerifierOutput,
)
from ..graph.builder import (
    CLAIM_TYPE_LIST,
    POLARITY_LIST,
    claim_features,
    build_torch_data,
)
from ..graph.rules import RuleViolation
from .base import BaseVerifier

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


_CLAIM_FEATURE_DIM = len(CLAIM_TYPE_LIST) + len(POLARITY_LIST) + 5
_NUM_HEADS = 4  # p_true, p_false, conflict, sufficiency. uncertainty derived.


def _to_features(graph: EvidenceGraph, claim_id: str) -> List[float]:
    return claim_features(graph, claim_id)


# ---------------------------------------------------------------------------
# MLP verifier
# ---------------------------------------------------------------------------


class _ClaimMLP(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    def __init__(self, in_dim: int = _CLAIM_FEATURE_DIM, hidden: int = 128):
        if not _HAS_TORCH:
            raise ImportError("torch is required for MLPClaimVerifier")
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_truth = nn.Linear(hidden, 2)           # logits over {true, false}
        self.head_conflict = nn.Linear(hidden, 1)
        self.head_sufficiency = nn.Linear(hidden, 1)

    def forward(self, x):  # type: ignore[override]
        h = self.backbone(x)
        return {
            "logits": self.head_truth(h),
            "conflict": torch.sigmoid(self.head_conflict(h)).squeeze(-1),
            "sufficiency": torch.sigmoid(self.head_sufficiency(h)).squeeze(-1),
        }


class MLPClaimVerifier(BaseVerifier):
    """Per-claim MLP head.

    The forward feature vector is the same one used by the rule verifier
    plus type / polarity one-hot; this keeps the two verifiers comparable
    and means the hybrid alpha-mixing is meaningful.
    """

    def __init__(self, model: Optional[_ClaimMLP] = None, device: str = "cpu"):
        if not _HAS_TORCH:
            raise ImportError("torch is required for MLPClaimVerifier")
        self.device = device
        self.model = model or _ClaimMLP()
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MLPClaimVerifier":
        if not _HAS_TORCH:
            raise ImportError("torch is required for MLPClaimVerifier")
        ckpt = torch.load(path, map_location=device)
        model = _ClaimMLP()
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        return cls(model=model, device=device)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)

    def verify(
        self,
        graph: EvidenceGraph,
        violations: Optional[List[RuleViolation]] = None,
    ) -> Dict[str, VerifierOutput]:
        if not graph.claims:
            return {}
        ids = list(graph.claims.keys())
        feats = torch.tensor(
            [_to_features(graph, cid) for cid in ids],
            dtype=torch.float, device=self.device,
        )
        with torch.no_grad():
            out = self.model(feats)
        probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        conflict = out["conflict"].cpu().numpy()
        sufficiency = out["sufficiency"].cpu().numpy()

        results: Dict[str, VerifierOutput] = {}
        for i, cid in enumerate(ids):
            p_true = float(probs[i, 0])
            p_false = float(probs[i, 1])
            results[cid] = VerifierOutput(
                claim_id=cid,
                p_true=p_true,
                p_false=p_false,
                uncertainty=float(_binary_entropy(p_true) + (1.0 - sufficiency[i])),
                conflict=float(conflict[i]),
                sufficiency=float(sufficiency[i]),
                rationale="mlp",
            )
        return results


# ---------------------------------------------------------------------------
# Graph transformer verifier
# ---------------------------------------------------------------------------


class _HeteroGraphTransformer(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """Hetero GAT over (claim, evidence, region) nodes with four edge types.

    Two-layer hetero attention, followed by a per-claim head that emits
    the same four quantities as the MLP verifier. Falls back to a simple
    bag-of-evidence pooling if torch_geometric is unavailable.
    """

    def __init__(
        self,
        claim_dim: int,
        evidence_dim: int,
        region_dim: int,
        hidden: int = 128,
        heads: int = 4,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch is required for GraphTransformerVerifier")
        super().__init__()
        try:
            from torch_geometric.nn import HeteroConv, GATConv  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch_geometric is required for GraphTransformerVerifier; "
                "install with `pip install torch-geometric`"
            ) from e
        from torch_geometric.nn import HeteroConv, GATConv

        self.proj_claim = nn.Linear(claim_dim, hidden)
        self.proj_evidence = nn.Linear(evidence_dim, hidden)
        self.proj_region = nn.Linear(region_dim, hidden)

        def _layer():
            return HeteroConv({
                ("evidence", "supports", "claim"):
                    GATConv((hidden, hidden), hidden // heads, heads=heads, add_self_loops=False),
                ("evidence", "contradicts", "claim"):
                    GATConv((hidden, hidden), hidden // heads, heads=heads, add_self_loops=False),
                ("evidence", "grounds", "region"):
                    GATConv((hidden, hidden), hidden // heads, heads=heads, add_self_loops=False),
                ("claim", "relates", "claim"):
                    GATConv(hidden, hidden // heads, heads=heads, add_self_loops=True),
            }, aggr="sum")

        self.conv1 = _layer()
        self.conv2 = _layer()

        self.head_truth = nn.Linear(hidden, 2)
        self.head_conflict = nn.Linear(hidden, 1)
        self.head_sufficiency = nn.Linear(hidden, 1)

    def forward(self, data):  # type: ignore[override]
        x = {
            "claim": self.proj_claim(data["claim"].x),
            "evidence": self.proj_evidence(data["evidence"].x),
            "region": self.proj_region(data["region"].x),
        }
        x = self.conv1(x, data.edge_index_dict)
        x = {k: torch.gelu(v) for k, v in x.items()}
        x = self.conv2(x, data.edge_index_dict)
        h = x["claim"]
        return {
            "logits": self.head_truth(h),
            "conflict": torch.sigmoid(self.head_conflict(h)).squeeze(-1),
            "sufficiency": torch.sigmoid(self.head_sufficiency(h)).squeeze(-1),
        }


class GraphTransformerVerifier(BaseVerifier):
    """Hetero graph transformer over the evidence graph.

    Build the PyG HeteroData on demand from the EvidenceGraph, run the
    learned model, then map the per-claim outputs back to claim ids.
    """

    def __init__(
        self,
        model: Optional["_HeteroGraphTransformer"] = None,
        device: str = "cpu",
        claim_dim: int = _CLAIM_FEATURE_DIM,
        evidence_dim: int = 13,
        region_dim: int = 10,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch is required for GraphTransformerVerifier")
        self.device = device
        self.model = model or _HeteroGraphTransformer(claim_dim, evidence_dim, region_dim)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "GraphTransformerVerifier":
        if not _HAS_TORCH:
            raise ImportError("torch is required for GraphTransformerVerifier")
        ckpt = torch.load(path, map_location=device)
        model = _HeteroGraphTransformer(**kwargs)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        return cls(model=model, device=device, **kwargs)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)

    def verify(
        self,
        graph: EvidenceGraph,
        violations: Optional[List[RuleViolation]] = None,
    ) -> Dict[str, VerifierOutput]:
        if not graph.claims:
            return {}
        data = build_torch_data(graph)
        if data is None:
            # torch_geometric not installed; should not happen because the
            # constructor would have raised. Defensive fallback to a flat
            # MLP forward over claim features only.
            return _flat_fallback(graph, self.device)
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data)
        probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        conflict = out["conflict"].cpu().numpy()
        sufficiency = out["sufficiency"].cpu().numpy()
        ids = list(graph.claims.keys())
        results: Dict[str, VerifierOutput] = {}
        for i, cid in enumerate(ids):
            p_true = float(probs[i, 0])
            p_false = float(probs[i, 1])
            results[cid] = VerifierOutput(
                claim_id=cid,
                p_true=p_true,
                p_false=p_false,
                uncertainty=float(_binary_entropy(p_true) + (1.0 - sufficiency[i])),
                conflict=float(conflict[i]),
                sufficiency=float(sufficiency[i]),
                rationale="graph_transformer",
            )
        return results


def _flat_fallback(graph: EvidenceGraph, device: str) -> Dict[str, VerifierOutput]:
    model = _ClaimMLP()
    model.to(device).eval()
    ids = list(graph.claims.keys())
    feats = torch.tensor(
        [_to_features(graph, cid) for cid in ids],
        dtype=torch.float, device=device,
    )
    with torch.no_grad():
        out = model(feats)
    probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
    conflict = out["conflict"].cpu().numpy()
    sufficiency = out["sufficiency"].cpu().numpy()
    results: Dict[str, VerifierOutput] = {}
    for i, cid in enumerate(ids):
        p_true = float(probs[i, 0])
        p_false = float(probs[i, 1])
        results[cid] = VerifierOutput(
            claim_id=cid, p_true=p_true, p_false=p_false,
            uncertainty=float(_binary_entropy(p_true) + (1.0 - sufficiency[i])),
            conflict=float(conflict[i]), sufficiency=float(sufficiency[i]),
            rationale="mlp_fallback",
        )
    return results


def _binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2)
