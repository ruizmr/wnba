"""Aggregation helpers for combining multiple lens outputs.

At MVP we only expose a simple geometric mean which has the nice property of
punishing values near 0 more harshly than the arithmetic mean.
Later we can replace this with a learned ensemble, but keeping the same
function *signature* means downstream code will not break.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = ["geo_mean"]

def geo_mean(values: Iterable[float]) -> float:
    """Return the geometric mean of *values*.

    Empty iterables yield 0 by definition here (no stake).
    All inputs are clamped to ≥ 0 to avoid math domain errors; callers should
    ensure they are already in [0, 1].
    """
    vals = list(values)
    if not vals:
        return 0.0
    product = 1.0
    for v in vals:
        product *= max(v, 0.0)
    return product ** (1 / len(vals))