"""
lenses.py
~~~~~~~~~~
Pure-function helpers that transform raw model predictions and betting market inputs into actionable
numbers such as expected value, Kelly stake fraction, and edges.  All functions are side-effect free
and can be unit-tested in isolation.

Example
-------
>>> from lenses import kelly_fraction
>>> kelly_fraction(win_prob=0.55, decimal_odds=1.91)
0.08343949044585987
"""
from __future__ import annotations

import math

__all__ = [
    "expected_value",
    "edge",
    "decimal_to_implied_prob",
    "kelly_fraction",
]


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Return the break-even probability for *decimal_odds*.

    Parameters
    ----------
    decimal_odds : float
        Market odds in the **decimal** format (e.g. 2.10 means risk 1 to win 1.10).

    Returns
    -------
    float
        Probability *p* such that :math:`p = 1 / O` where *O* is the decimal odds.

    Raises
    ------
    ValueError
        If ``decimal_odds`` is not strictly greater than *1.0*.
    """
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0 to carry positive return potential.")
    return 1.0 / decimal_odds


def expected_value(win_prob: float, decimal_odds: float) -> float:
    """Expected profit per unit stake given a win probability and market odds.

    ``expected_value`` is defined as::

        EV = p * (O - 1) - (1 - p)

    where *p* is the bettor's estimated win probability and *O* is the decimal odds.

    A positive EV indicates a +EV wager.
    """
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError("win_prob must lie in [0, 1].")
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0.")

    b = decimal_odds - 1.0  # net odds (profit per unit risked)
    return win_prob * b - (1.0 - win_prob)


def edge(win_prob: float, decimal_odds: float) -> float:
    """Return the fractional *edge* over the sportsbook implied probability.

    Edge is defined as::

        edge = p - p_implied

    where *p* is our win probability and *p_implied* = 1 / O.
    """
    p_implied = decimal_to_implied_prob(decimal_odds)
    return win_prob - p_implied


def kelly_fraction(
    win_prob: float,
    decimal_odds: float,
    *,
    multiplier: float = 1.0,
    max_fraction: float | None = 1.0,
) -> float:
    """Compute the Kelly bet fraction for a single binary-outcome wager.

    The canonical Kelly criterion (Kelly, 1956) maximises the expected logarithmic growth
    of capital.  For decimal odds O and subjective win probability p, the optimal fraction
    of bankroll to wager is::

        f* = (p * (O - 1) - (1 - p)) / (O - 1)
            = (p * O - 1) / (O - 1)

    The function returns *max(0, f*)* and optionally caps the value at ``max_fraction``.

    Parameters
    ----------
    win_prob : float
        Subjective probability of winning (0 ≤ p ≤ 1).
    decimal_odds : float
        Market odds in **decimal** format (> 1.0).
    multiplier : float, default 1.0
        Scaling factor for the optimal Kelly stake.
    max_fraction : float | None, default 1.0
        Upper bound on the Kelly fraction, useful for *fractional Kelly* strategies
        (e.g., set to 0.5 for *half-Kelly*).  Pass ``None`` to disable the cap.

    Returns
    -------
    float
        Fraction of current bankroll to stake.  0.0 implies *no bet*.
    """
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError("win_prob must lie in [0, 1].")
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be > 1.0.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")
    if max_fraction is not None and max_fraction <= 0:
        raise ValueError("max_fraction must be positive or None.")

    numerator = win_prob * decimal_odds - 1.0
    denominator = decimal_odds - 1.0
    f_star = numerator / denominator

    # Truncate at zero (no negative staking) and apply user-defined scaling and cap.
    f_star = max(0.0, f_star) * multiplier
    if max_fraction is not None:
        f_star = min(f_star, max_fraction)

    return f_star