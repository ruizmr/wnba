"""Loss functions for model training.

Currently includes:
- CalibratedKellyLoss: Kelly-aware calibration objective.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["CalibratedKellyLoss"]


class CalibratedKellyLoss(nn.Module):
    """Kelly-aware loss encouraging probabilistic calibration.

    Given predicted winning probabilities ``p`` and decimal odds ``o`` the
    Kelly-optimal log growth is :math:`\log(p \cdot o - p + 1)` when betting
    the Kelly fraction.  We **minimise the negative** of the expected growth
    (multiplied by \(\lambda_\text{kelly}\)) so that lower loss => higher
    expected bankroll growth.

    Notes
    -----
    • Inputs are assumed to be *probabilities* in \[0,1\] **after** temperature
      scaling; logits should be converted prior to calling the loss.
    • ``decimal_odds`` must be ≥ 1.
    """

    def __init__(self, lambda_kelly: float = 0.1):
        super().__init__()
        if lambda_kelly < 0:
            raise ValueError("lambda_kelly must be non-negative")
        self.lambda_kelly = lambda_kelly

    def forward(self, prob: torch.Tensor, decimal_odds: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if prob.shape != decimal_odds.shape:
            raise ValueError("prob and decimal_odds must have the same shape")
        if torch.any(decimal_odds < 1):
            raise ValueError("decimal odds must be >= 1.0")

        growth = torch.log(prob * decimal_odds - prob + 1)  # Kelly log growth
        loss = -self.lambda_kelly * growth.mean()
        return loss