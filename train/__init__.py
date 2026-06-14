"""Training utilities for EGA-MedRAX verifiers.

We keep this ``__init__`` lazy so that importing ``train.pseudo_label``
(which is torch-free) does not drag in torch, torch_geometric, etc. The
heavy modules are still importable directly:

    from train.losses import EGALoss
    from train.dataset import EGAGraphDataset
    from train.pseudo_label import build_pseudo_dataset
"""

__all__ = [
    "build_pseudo_dataset",
    "EGALoss",
    "claim_loss",
    "conflict_loss",
    "calibration_loss",
    "abstention_loss",
    "EGAGraphDataset",
]


def __getattr__(name):
    if name == "build_pseudo_dataset":
        from .pseudo_label import build_pseudo_dataset
        return build_pseudo_dataset
    if name in {"EGALoss", "claim_loss", "conflict_loss",
                "calibration_loss", "abstention_loss"}:
        from . import losses
        return getattr(losses, name)
    if name == "EGAGraphDataset":
        from .dataset import EGAGraphDataset
        return EGAGraphDataset
    raise AttributeError(name)
