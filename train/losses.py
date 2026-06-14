"""Training losses for the EGA verifier.

L = L_claim + lambda_conflict * L_conflict
              + lambda_calib    * L_calib
              + lambda_abstain  * L_abstain

* claim_loss:    cross-entropy on per-claim {true, false, unknown} labels
* conflict_loss: BCE on whether the evidence set is internally conflicting
* calibration_loss: Brier score on the predicted truth probability
* abstention_loss: selective-prediction loss after Geifman & El-Yaniv (2017)
                   coverage * (avg_loss_on_covered) + lambda * (target_cov - cov)^2

The selective-prediction term is what gives us the "answer correctly on
covered samples, abstain on uncertain ones" objective from the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------


def claim_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over {true=0, false=1}.

    Unknown claims are masked with label = -1 and contribute 0.
    """
    mask = labels >= 0
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], labels[mask].long())


def conflict_loss(conflict_pred: torch.Tensor, conflict_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(conflict_pred.clamp(1e-6, 1 - 1e-6), conflict_target.float())


def calibration_loss(p_true: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Brier score on the binary target (labels in {0, 1}, -1 ignored)."""
    mask = labels >= 0
    if mask.sum() == 0:
        return p_true.new_zeros(())
    target = (labels[mask] == 0).float()  # label 0 == "true"
    return torch.mean((p_true[mask] - target) ** 2)


def abstention_loss(
    p_true: torch.Tensor,
    labels: torch.Tensor,
    sufficiency: torch.Tensor,
    target_coverage: float = 0.8,
    lam: float = 16.0,
) -> torch.Tensor:
    """Selective prediction loss (Geifman & El-Yaniv).

    The model controls coverage through ``sufficiency``: high sufficiency
    means we are willing to commit to the prediction, low sufficiency
    means we abstain. We push the soft coverage toward ``target_coverage``
    while minimising loss on covered samples.
    """
    mask = labels >= 0
    if mask.sum() == 0:
        return p_true.new_zeros(())
    target = (labels[mask] == 0).float()
    per_sample = (p_true[mask] - target) ** 2
    cov = sufficiency[mask]
    coverage = cov.mean().clamp(min=1e-3)
    covered_loss = (per_sample * cov).sum() / cov.sum().clamp(min=1e-3)
    coverage_penalty = lam * F.relu(target_coverage - coverage) ** 2
    return covered_loss + coverage_penalty


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


@dataclass
class EGALoss:
    lambda_conflict: float = 0.5
    lambda_calib: float = 0.5
    lambda_abstain: float = 0.3
    target_coverage: float = 0.8

    def __call__(
        self,
        logits: torch.Tensor,
        conflict_pred: torch.Tensor,
        sufficiency_pred: torch.Tensor,
        labels: torch.Tensor,
        conflict_target: torch.Tensor,
    ) -> dict:
        p_true = torch.softmax(logits, dim=-1)[:, 0]
        loss_c = claim_loss(logits, labels)
        loss_conf = conflict_loss(conflict_pred, conflict_target)
        loss_calib = calibration_loss(p_true, labels)
        loss_abs = abstention_loss(p_true, labels, sufficiency_pred, self.target_coverage)
        total = (
            loss_c
            + self.lambda_conflict * loss_conf
            + self.lambda_calib * loss_calib
            + self.lambda_abstain * loss_abs
        )
        return {
            "total": total,
            "claim": loss_c.detach(),
            "conflict": loss_conf.detach(),
            "calibration": loss_calib.detach(),
            "abstention": loss_abs.detach(),
        }
