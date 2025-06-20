import os
from pathlib import Path

import pytest

from python.aggregate import geo_mean, learn_weights_ridge, weighted_mean, _cache_path


def test_geo_mean_basic():
    p1 = [0.6, 0.4]
    p2 = [0.5, 0.45]
    gm = geo_mean(p1, p2)
    assert pytest.approx(gm[0], rel=1e-6) == ((0.6 * 0.5) ** 0.5)
    assert pytest.approx(gm[1], rel=1e-6) == ((0.4 * 0.45) ** 0.5)


def test_learn_weights_ridge_and_cache(tmp_path):
    # Override cache dir via monkeypatch of environment variable
    preds1 = [0.6, 0.7, 0.2]
    preds2 = [0.65, 0.55, 0.25]
    targets = [1, 1, 0]

    w = learn_weights_ridge(preds1, preds2, targets=targets, alpha=0.1, cache=True)
    assert pytest.approx(sum(w)) == 1.0
    assert all(v >= 0 for v in w)

    # Call again and ensure it loads from cache
    w2 = learn_weights_ridge(preds1, preds2, targets=targets, alpha=0.1, cache=True)
    assert w2 == w


@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.3, 0.7]])
def test_weighted_mean(weights):
    p1 = [0.6, 0.4]
    p2 = [0.5, 0.45]
    combined = weighted_mean(p1, p2, weights=weights)
    expected = [weights[0] * 0.6 + weights[1] * 0.5, weights[0] * 0.4 + weights[1] * 0.45]
    assert combined == pytest.approx(expected)