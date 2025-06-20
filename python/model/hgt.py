"""Minimal stand-in for a Heterogeneous Graph Transformer.

The real implementation will rely on **torch-geometric** but for the purpose of
unit tests and CI we use a simple `nn.Linear` projection while preserving the
public interface described in Edge-29.
"""
from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["MiniHGT"]


def _reset_parameters(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)


class MiniHGT(nn.Module):
    """A **very** small placeholder for the future HGT.

    It supports temperature scaling and the `return_logits` flag so that
    downstream code can be developed now and swapped later.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # Temperature parameter initialised at 1.0 (log-space = 0)
        self.calib_temperature = nn.Parameter(torch.zeros(1))

        _reset_parameters(self)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *, return_logits: bool = False):  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x
            Node features tensor of shape *(B, in_dim)*.
        return_logits
            When *True*, returns raw logits (pre-sigmoid).  Otherwise returns
            calibrated probabilities in *[0,1]*.
        """
        logits = self.proj(x)
        if return_logits:
            return logits
        # Apply temperature scaling: prob = sigmoid(logits / exp(T)) where T is log temp param
        prob = torch.sigmoid(logits / self.calib_temperature.exp())
        return prob