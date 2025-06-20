"""losses.py
Utility functions and composite loss classes for model training.

Core idea
=========
We want to learn probabilities `p_hat` that are simultaneously
(1) well‐calibrated (cross-entropy / negative log-likelihood) and (2) lead to
optimal Kelly wagering when combined with the sportsbook's decimal odds.

Loss(p, y, o) = CE(p, y)  –  λ * log_growth(p, y, o)

where
    CE(p, y) =  - [ y * log(p) + (1-y) * log(1-p) ]
    log_growth = log(1 + k * (o-1))   if y == 1
               = log(1 - k)           if y == 0
    k = Kelly fraction = (p * o - 1) / (o - 1)

λ (lambda_kelly) trades off calibration vs bankroll growth; λ ∈ [0,1].

Clipping:   p ∈ (eps, 1-eps) to avoid NaN; k is clipped to [-k_max, k_max]
with k_max < 1 to respect no-short-selling & risk control; default k_max=0.5.

Example (PyTorch):
    criterion = CalibratedKellyLoss(lambda_kelly=0.05)
    loss = criterion(p_hat, target, odds)

The class is fully differentiable, allowing gradient descent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class CalibratedKellyLoss(torch.nn.Module):
    """Cross-entropy + λ * (negative) Kelly log-wealth objective.

    Parameters
    ----------
    lambda_kelly : float, optional
        Weight on the Kelly term. 0 ⇒ pure cross-entropy, 1 ⇒ maximise log-growth
        (note the sign is negative in loss), by default 0.05.
    eps : float, optional
        Numerical stability for clipping probabilities, by default 1e-7.
    k_max : float, optional
        Cap on absolute Kelly fraction to avoid overbetting, by default 0.5.
    """

    def __init__(self, lambda_kelly: float = 0.05, eps: float = 1e-7, k_max: float = 0.5):
        super().__init__()
        assert 0 <= lambda_kelly <= 1, "lambda_kelly must be in [0,1]"
        self.lambda_kelly = lambda_kelly
        self.eps = eps
        self.k_max = k_max

    def forward(self, p_hat: torch.Tensor, target: torch.Tensor, odds: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute batch loss.

        Parameters
        ----------
        p_hat : torch.Tensor
            Model predicted win probability, shape (N,) or (N,1)
        target : torch.Tensor
            Ground-truth outcome {0,1}, shape broadcastable to p_hat.
        odds : torch.Tensor
            Decimal odds offered for the positive class (i.e. payout for a 1-unit
            stake if the outcome = 1). Shape broadcastable to p_hat.
        """
        # ensure tensors are 1-D
        p_hat = p_hat.squeeze()
        target = target.squeeze().float()
        odds = odds.squeeze().float()

        # Clip probabilities
        p_hat = torch.clamp(p_hat, self.eps, 1 - self.eps)

        # ----- Cross-entropy (log loss) -----
        ce = F.binary_cross_entropy(p_hat, target, reduction="none")  # shape (N,)

        # ----- Kelly term -----
        kelly_fraction = (p_hat * odds - 1) / (odds - 1)
        kelly_fraction = torch.clamp(kelly_fraction, -self.k_max, self.k_max)

        gain_if_win = torch.log1p(kelly_fraction * (odds - 1))  # log(1 + k*(o-1))
        loss_if_lose = torch.log1p(-kelly_fraction)  # log(1 - k)
        # realised log growth for the sample depending on outcome
        log_growth = target * gain_if_win + (1 - target) * loss_if_lose

        # We *maximize* log_growth, so negative for loss.
        kelly_term = -log_growth  # shape (N,)

        # ----- Combine -----
        combined = ce + self.lambda_kelly * kelly_term
        return combined.mean()