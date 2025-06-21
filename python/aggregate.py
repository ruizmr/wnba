"""
aggregate.py
~~~~~~~~~~~~
Model ensembling utilities.

Functions
---------
geo_mean
    Compute the geometric mean of probabilities across multiple models.
learn_weights_ridge
    Learn non-negative Ridge-regularised weights that minimise squared error to
    a target vector, then normalise to sum to 1.
weighted_mean
    Combine model predictions using learned weights.

A lightweight *file-based* cache avoids re-fitting if the training data hash
has not changed.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable, Sequence, List, Union

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["geo_mean", "learn_weights_ridge", "weighted_mean", "_cache_path"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    """Return a deterministic cache path for *key*."""
    return CACHE_DIR / f"ridge_{key}.pkl"


def _hash_arrays(arrays: Sequence[Sequence[float]]) -> str:
    """Deterministically hash *arrays* for cache invalidation."""
    m = hashlib.sha256()
    for arr in arrays:
        m.update(json.dumps(list(arr), sort_keys=True).encode())
    return m.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geo_mean(values: Union[List[float], List[List[float]]]) -> float | List[float]:
    """Geometric mean helper with ergonomic behaviour for *both* scalar and
    vector use-cases.

    Rules
    -----
    1. If **values** is a *flat* list (``[0.1, 0.4]``) → return a **scalar**.
    2. If **values** is an *iterator of iterables* (``[[0.6,0.4],[0.5,0.45]]``)
       → return an element-wise list like the classic ensemble geo-mean.
    3. Empty input returns ``0.0`` to satisfy edge-case tests.
    """

    values_list = list(values)  # type: ignore[arg-type]
    if not values_list:
        return 0.0

    first = values_list[0]

    # Case 1 ‑ flat list of floats
    if isinstance(first, (int, float)):
        prod = 1.0
        for v in values_list:  # type: ignore[arg-type]
            prod *= float(v)
        return prod ** (1.0 / len(values_list))

    # Case 2 ‑ list of iterables → ensemble element-wise geo mean
    vectors = [list(map(float, vec)) for vec in values_list]  # type: ignore[arg-type]
    n_models = len(vectors)
    n_samples = len(vectors[0])
    if any(len(v) != n_samples for v in vectors):
        raise ValueError("All probability arrays must be the same length.")

    out: list[float] = []
    for i in range(n_samples):
        prod = 1.0
        for vec in vectors:
            prod *= vec[i]
        out.append(prod ** (1.0 / n_models))
    return out


def learn_weights_ridge(
    *preds: Iterable[float],
    targets: Iterable[float],
    alpha: float = 1.0,
    cache: bool = True,
) -> list[float]:
    """Learn ensemble weights via closed-form Ridge regression.

    Parameters
    ----------
    *preds : Iterable[float]
        Variable number of prediction vectors (each len == n_samples).
    targets : Iterable[float]
        Ground-truth probabilities (len == n_samples).
    alpha : float, default 1.0
        Ridge regularisation strength.
    cache : bool, default True
        Cache weights on disk using a content hash of inputs.

    Returns
    -------
    list[float]
        Normalised non-negative weights summing to 1.
    """
    import math

    X = [list(p) for p in preds]
    y = list(targets)

    if not X:
        raise ValueError("At least one prediction array is required.")
    n_models = len(X)
    n_samples = len(y)
    if any(len(p) != n_samples for p in X):
        raise ValueError("All prediction arrays must have same length as targets.")
    if n_samples == 0:
        raise ValueError("Targets must not be empty.")

    key = _hash_arrays(X + [y])
    path = _cache_path(key)
    if cache and path.exists():
        return pickle.loads(path.read_bytes())

    # Build normal equations X^T X + alpha I  and X^T y.
    # Since n_models is small, do naive loops without numpy.
    XtX = [[0.0] * n_models for _ in range(n_models)]
    Xty = [0.0] * n_models
    for i in range(n_models):
        for j in range(n_models):
            s = 0.0
            for k in range(n_samples):
                s += X[i][k] * X[j][k]
            if i == j:
                s += alpha
            XtX[i][j] = s
        s = 0.0
        for k in range(n_samples):
            s += X[i][k] * y[k]
        Xty[i] = s

    # Solve linear system (XtX) w = Xty via Gaussian elimination.
    # Augment matrix.
    augmented = [row + [val] for row, val in zip(XtX, Xty)]
    n = n_models
    for i in range(n):
        # Pivot.
        pivot = augmented[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix; try larger alpha")
        factor = pivot
        for j in range(i, n + 1):
            augmented[i][j] /= factor
        # Eliminate below.
        for r in range(n):
            if r == i:
                continue
            factor = augmented[r][i]
            for c in range(i, n + 1):
                augmented[r][c] -= factor * augmented[i][c]

    w = [augmented[i][-1] for i in range(n)]

    # Normalise & clip negatives to 0
    w = [max(0.0, v) for v in w]
    s = sum(w)
    if s == 0.0:
        # Fallback equal weights
        w = [1.0 / n_models] * n_models
    else:
        w = [v / s for v in w]

    if cache:
        path.write_bytes(pickle.dumps(w))
    return w


def weighted_mean(*preds: Iterable[float], weights: Sequence[float]) -> list[float]:
    """Return weighted mean of predictions using *weights*."""
    weights = list(weights)
    X = [list(p) for p in preds]
    if not X:
        raise ValueError("At least one prediction array required")
    n_samples = len(X[0])
    if any(len(p) != n_samples for p in X):
        raise ValueError("All prediction arrays must be same length")
    if len(weights) != len(X):
        raise ValueError("Number of weights must match predictions")
    out: list[float] = []
    for i in range(n_samples):
        s = 0.0
        for w, p in zip(weights, X):
            s += w * p[i]
        out.append(s)
    return out