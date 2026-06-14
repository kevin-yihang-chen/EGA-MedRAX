"""Train the EGA verifier on the pseudo-labelled graph dataset.

Defaults reflect the design doc's Stage 2 plan:

  * losses : claim CE + conflict BCE + Brier + selective-prediction
  * model  : either MLPClaimVerifier (cheap, version-1) or
             GraphTransformerVerifier (version-2, main-conference path)
  * schedule : AdamW + cosine + 100 epochs on the synthetic set

Example
-------
    python -m train.train_verifier \
        --data data/ega_pseudo \
        --model graph \
        --output checkpoints/ega_v2.pt

Stage 3 (verifier-driven tool acquisition) lives in the agent itself
through the refine node; this script trains the static verifier only.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ega_medrax.graph.builder import (
    CLAIM_TYPE_LIST,
    POLARITY_LIST,
    EVIDENCE_TYPE_LIST,
    PROPOSITION_LIST,
)
from ega_medrax.verifier.learned_verifier import (
    GraphTransformerVerifier,
    MLPClaimVerifier,
    _CLAIM_FEATURE_DIM,
    _ClaimMLP,
    _HeteroGraphTransformer,
)

from .dataset import EGAGraphDataset
from .losses import EGALoss


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def _collate_flat(batch):
    flat = torch.cat([b["flat_features"] for b in batch], dim=0)
    labels = torch.cat([b["claim_labels"] for b in batch], dim=0)
    conflict = torch.cat([b["conflict_labels"] for b in batch], dim=0)
    return flat, labels, conflict


def _collate_graph(batch):
    # PyG Batch handles graph batching when torch_geometric is installed.
    from torch_geometric.data import Batch
    datas = [b["graph_data"] for b in batch if b["graph_data"] is not None]
    if not datas:
        return None, None, None
    batched = Batch.from_data_list(datas)
    labels = torch.cat([b["claim_labels"] for b in batch
                        if b["graph_data"] is not None], dim=0)
    conflict = torch.cat([b["conflict_labels"] for b in batch
                          if b["graph_data"] is not None], dim=0)
    return batched, labels, conflict


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_mlp(args) -> None:
    dataset = EGAGraphDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate_flat)
    model = _ClaimMLP(in_dim=_CLAIM_FEATURE_DIM, hidden=args.hidden).to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = EGALoss()

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for epoch in range(args.epochs):
        running = defaultdict(float)
        for flat, labels, conflict in loader:
            flat = flat.to(args.device); labels = labels.to(args.device); conflict = conflict.to(args.device)
            out = model(flat)
            losses = loss_fn(out["logits"], out["conflict"], out["sufficiency"], labels, conflict)
            opt.zero_grad(); losses["total"].backward(); opt.step()
            for k, v in losses.items():
                running[k] += float(v.detach())
        sched.step()
        _log_epoch(epoch, running, len(loader))

    MLPClaimVerifier(model=model, device=args.device).save(args.output)
    logger.info("saved MLP verifier to %s", args.output)


def train_graph(args) -> None:
    dataset = EGAGraphDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate_graph)
    claim_dim = _CLAIM_FEATURE_DIM
    evidence_dim = len(EVIDENCE_TYPE_LIST) + len(PROPOSITION_LIST) + 4
    region_dim = 3 + 3 + 4
    model = _HeteroGraphTransformer(
        claim_dim=claim_dim, evidence_dim=evidence_dim,
        region_dim=region_dim, hidden=args.hidden, heads=4,
    ).to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = EGALoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        running = defaultdict(float)
        for batched, labels, conflict in loader:
            if batched is None:
                continue
            batched = batched.to(args.device); labels = labels.to(args.device); conflict = conflict.to(args.device)
            out = model(batched)
            losses = loss_fn(out["logits"], out["conflict"], out["sufficiency"], labels, conflict)
            opt.zero_grad(); losses["total"].backward(); opt.step()
            for k, v in losses.items():
                running[k] += float(v.detach())
        sched.step()
        _log_epoch(epoch, running, len(loader))

    GraphTransformerVerifier(
        model=model, device=args.device,
        claim_dim=claim_dim, evidence_dim=evidence_dim, region_dim=region_dim,
    ).save(args.output)
    logger.info("saved graph verifier to %s", args.output)


def _log_epoch(epoch: int, running: Dict[str, float], n: int) -> None:
    if n == 0:
        return
    msg = ", ".join(f"{k}={v / n:.4f}" for k, v in running.items())
    logger.info("epoch %d | %s", epoch, msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dir of pseudo-labelled JSONs")
    parser.add_argument("--model", choices=["mlp", "graph"], default="mlp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    (train_graph if args.model == "graph" else train_mlp)(args)


if __name__ == "__main__":
    main()
