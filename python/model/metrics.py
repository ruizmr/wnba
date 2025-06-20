"""Custom metrics independent of torchmetrics external dependency."""
from __future__ import annotations

import torch

__all__ = ["ExpectedCalibrationError", "BrierScore"]


class _BaseMetric:  # minimal interface
    def __init__(self):
        self.reset()

    def update(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError

    def compute(self):  # noqa: D401
        raise NotImplementedError

    def reset(self):  # noqa: D401
        raise NotImplementedError


class ExpectedCalibrationError(_BaseMetric):
    """Expected Calibration Error (ECE) with equal-width probability bins.

    Implementation follows the 15-bin variant used in many calibration papers.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins
        self.reset()

    def reset(self):  # noqa: D401
        self._conf_sum = torch.zeros(self.n_bins)
        self._acc_sum = torch.zeros(self.n_bins)
        self._count = torch.zeros(self.n_bins)

    def update(self, prob: torch.Tensor, target: torch.Tensor):  # noqa: D401
        prob = prob.detach().flatten()
        target = target.detach().flatten().to(prob.dtype)
        bin_ids = torch.clamp((prob * self.n_bins).long(), max=self.n_bins - 1)
        for b in range(self.n_bins):
            mask = bin_ids == b
            if mask.any():
                self._count[b] += mask.sum()
                self._conf_sum[b] += prob[mask].sum()
                self._acc_sum[b] += target[mask].sum()

    def compute(self):  # noqa: D401
        nonzero = self._count > 0
        if not nonzero.any():
            return torch.tensor(0.0)
        conf_avg = self._conf_sum[nonzero] / self._count[nonzero]
        acc_avg = self._acc_sum[nonzero] / self._count[nonzero]
        ece = torch.sum(torch.abs(conf_avg - acc_avg) * (self._count[nonzero] / self._count.sum()))
        return ece


class BrierScore(_BaseMetric):
    """Brier score (mean squared error between probability and outcome)."""

    def reset(self):  # noqa: D401
        self._sum_sq: float = 0.0
        self._n: int = 0

    def update(self, prob: torch.Tensor, target: torch.Tensor):  # noqa: D401
        diff = (prob.flatten() - target.to(prob.dtype).flatten()) ** 2
        self._sum_sq += diff.sum().item()
        self._n += diff.numel()

    def compute(self):  # noqa: D401
        if self._n == 0:
            return torch.tensor(0.0)
        return torch.tensor(self._sum_sq / self._n)