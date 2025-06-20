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