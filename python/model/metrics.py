"""metrics.py
Utility functions to evaluate profitability of probabilistic betting models.
All functions operate on a 1-D torch / numpy array of *period* returns (not cumulative).
A period could be a single wager outcome or a daily portfolio return.
"""
from __future__ import annotations

import numpy as np


def _to_np(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float)


def sharpe_ratio(returns, risk_free: float = 0.0):
    r = _to_np(returns)
    excess = r - risk_free
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std()


def sortino_ratio(returns, risk_free: float = 0.0):
    r = _to_np(returns)
    downside = np.minimum(0, r - risk_free)
    denom = np.sqrt((downside ** 2).mean())
    if denom == 0:
        return 0.0
    return (r.mean() - risk_free) / denom


def max_drawdown(cumulative_returns):
    c = _to_np(cumulative_returns)
    peaks = np.maximum.accumulate(c)
    drawdowns = (c - peaks) / peaks
    return drawdowns.min()