"""Loss & metric helpers inspired by Big-Brain research ledger.

Functions
---------
calibrated_bce_kelly(logits, targets, odds, temperature)
    Binary-cross-entropy calibrated by learnable temperature *and* Kelly objective.
brier_score(probs, targets)
    Probability calibration error (squared diff).
ece(probs, targets, n_bins=10)
    Expected Calibration Error (ECE).
"""
from __future__ import annotations

from typing import Tuple

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for losses. Install via `pip install torch` or activate env.yml."
    ) from exc

__all__ = [
    "TemperatureScaler",
    "brier_score",
    "ece",
    "calibrated_bce_kelly",
    "CalibratedKellyLoss",
]


class TemperatureScaler(torch.nn.Module):
    """Single scalar temperature parameter for post-hoc calibration (Platt/Guo).

    Forward divides incoming *logits* by *T* (softmax temperature).
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.log(torch.tensor(init_temp)))

    @property
    def temperature(self) -> torch.Tensor:  # noqa: D401
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return logits / self.temperature


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Mean squared error between predicted probabilities and binary labels."""
    return torch.mean((probs - targets.float()) ** 2)


def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> torch.Tensor:  # noqa: D401
    """Expected Calibration Error (scalar)."""
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece_val = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.any():
            acc = targets[mask].float().mean()
            conf = probs[mask].mean()
            ece_val += (conf - acc).abs() * mask.float().mean()
    return ece_val

# -----------------------------------------------------------------------------
# Kelly-aware calibrated BCE loss
# -----------------------------------------------------------------------------

def _kelly_objective(probs: torch.Tensor, targets: torch.Tensor, odds: torch.Tensor) -> torch.Tensor:
    """Simplified binary Kelly criterion (expected log-wealth)."""
    # convert American odds to decimal odds if needed; assume decimal for now
    wealth = probs * torch.log1p((odds - 1) * targets) + (1 - probs) * torch.log1p(-probs)
    return -torch.mean(wealth)  # negative for minimisation


def calibrated_bce_kelly(
    logits: torch.Tensor,
    targets: torch.Tensor,
    odds: torch.Tensor,
    temperature: torch.nn.Parameter,
    alpha: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D401
    """Composite loss: calibrated BCE + Kelly objective.

    Parameters
    ----------
    logits : raw model outputs (before softmax)
    targets : binary labels (0/1)
    odds : decimal odds for Kelly term
    temperature : learnable scalar T
    alpha : weight on BCE (1-alpha on Kelly)
    """

    calibrated_logits = logits / temperature
    bce = F.binary_cross_entropy_with_logits(calibrated_logits, targets.float())
    probs = torch.sigmoid(calibrated_logits)
    kelly = _kelly_objective(probs, targets, odds)
    loss = alpha * bce + (1 - alpha) * kelly
    return loss, bce.detach(), kelly.detach()


class CalibratedKellyLoss(torch.nn.Module):
    """Cross‐entropy + λ * (negative) Kelly log‐wealth objective.

    Parameters
    ----------
    lambda_kelly : float, optional
        Weight on the Kelly term. 0 ⇒ pure cross‐entropy, 1 ⇒ maximise log‐growth,
        by default 0.05.
    eps : float, optional
        Numerical stability for clipping probabilities, by default 1e-7.
    k_max : float, optional
        Cap on absolute Kelly fraction to avoid overbetting, by default 0.5.
    """

    def __init__(self, lambda_kelly: float = 0.05, eps: float = 1e-7, k_max: float = 0.5):
        super().__init__()
        if not (0 <= lambda_kelly <= 1):
            raise ValueError("lambda_kelly must be in [0, 1]")
        self.lambda_kelly = lambda_kelly
        self.eps = eps
        self.k_max = k_max

    def forward(self, p_hat: torch.Tensor, target: torch.Tensor, odds: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # ensure tensors broadcast and are floats
        p_hat = p_hat.squeeze()
        target = target.squeeze().float()
        odds = odds.squeeze().float()

        # Probabilities clipping for numerical stability
        p_hat = torch.clamp(p_hat, self.eps, 1 - self.eps)

        # Cross-entropy
        ce = F.binary_cross_entropy(p_hat, target, reduction="none")

        # Kelly fraction (clipped)
        kelly_fraction = (p_hat * odds - 1) / (odds - 1)
        kelly_fraction = torch.clamp(kelly_fraction, -self.k_max, self.k_max)

        gain_if_win = torch.log1p(kelly_fraction * (odds - 1))  # log(1 + k*(o-1))
        loss_if_lose = torch.log1p(-kelly_fraction)  # log(1 - k)
        log_growth = target * gain_if_win + (1 - target) * loss_if_lose

        kelly_term = -log_growth  # maximise log_growth => minimise negative

        loss = ce + self.lambda_kelly * kelly_term
        return loss.mean()
