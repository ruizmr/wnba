"""Stake-sizing "lenses" applied to raw model edge predictions.

Each lens is a pure function: it *only* consumes its inputs and returns a
`float` between 0 and 1 representing the fraction of bankroll to stake.
That makes them straightforward to unit-test and to compose.

All lenses assume the same minimal input schema for now; later we can switch
these to Pydantic models once Agent 1 publishes the canonical `schema.py`.

The goal at the moment is to publish *working* reference implementations that
other agents can call while we iterate on the math.
"""
from __future__ import annotations

from typing import Iterable, Union

Number = Union[int, float]

__all__ = [
    "kelly_criterion",
    "sharpe_adjusted",
    "volatility_cap",
    "underdog_boost",
    "market_neutral",
]

def _clamp(value: Number, low: Number = 0.0, high: Number = 1.0) -> float:
    """Clamp *value* into the inclusive range [`low`, `high`]."""
    return float(max(low, min(high, value)))

def kelly_criterion(prob: float, odds: float, bankroll: float = 1.0) -> float:
    """Vanilla Kelly criterion.

    Parameters
    ----------
    prob
        Model-estimated win probability (0 – 1).
    odds
        Decimal odds offered by the market (e.g. *2.40* means +140 underdog).
    bankroll
        Current bankroll size (monetary units). Included so callers can scale
        into currency if they wish; for pure ratio outputs set to *1*.

    Returns
    -------
    float
        Fraction of *bankroll* to stake; clamped to [0, 1].

    Example
    -------
    >>> kelly_criterion(prob=0.55, odds=2.0)
    0.1
    """
    edge = prob * (odds - 1) - (1 - prob)
    if edge <= 0:
        return 0.0
    fraction = edge / (odds - 1)
    return bankroll * _clamp(fraction)

def sharpe_adjusted(
    kelly_fraction: float,
    sharpe_ratio: float,
    target_sharpe: float = 3.0,
) -> float:
    """Reduce Kelly stake when the underlying model has a low Sharpe ratio.

    The adjustment is linear: when *sharpe_ratio* = *target_sharpe*, the full
    *kelly_fraction* is returned; when the Sharpe drops to 0 the lens returns 0.
    """
    factor = _clamp(sharpe_ratio / target_sharpe)
    return kelly_fraction * factor

def volatility_cap(stake: float, max_fraction: float = 0.05) -> float:
    """Hard-cap the stake to *max_fraction* of bankroll."""
    return _clamp(min(stake, max_fraction))

def underdog_boost(stake: float, odds: float, threshold: float = 2.5, boost: float = 1.25) -> float:
    """Boost stake by *boost*× for longshots with odds ≥ *threshold*.

    The result is still clamped to ≤ 1.
    """
    if odds >= threshold:
        stake *= boost
    return _clamp(stake)

def market_neutral(stakes: Iterable[float]) -> float:
    """Return the mean absolute stake to keep total exposure ≈ 0.

    This lens is meant to be called *after* long/short pairing logic; here we
    simply scale the vector so that the mean absolute value equals the desired
    individual stake, returning that common stake.
    """
    stakes = list(stakes)
    if not stakes:
        return 0.0
    mean_abs = sum(abs(s) for s in stakes) / len(stakes)
    return _clamp(mean_abs)