"""Unit tests for `python.lenses`.

These tests exercise simple happy-path cases to guard against regressions as
we iterate on the math. They deliberately avoid asserting *exact* values so we
retain freedom to tweak the formulas without rewriting a bunch of tests.
"""
from python.lenses import (
    kelly_criterion,
    market_neutral,
    sharpe_adjusted,
    underdog_boost,
    volatility_cap,
)


def test_kelly_positive_edge():
    stake = kelly_criterion(prob=0.55, odds=2.0)
    assert 0 < stake <= 1


def test_kelly_no_edge_returns_zero():
    stake = kelly_criterion(prob=0.40, odds=1.8)
    assert stake == 0.0


def test_sharpe_adjustment_monotonic():
    base = 0.2
    s1 = sharpe_adjusted(base, sharpe_ratio=1.0, target_sharpe=2.0)
    s2 = sharpe_adjusted(base, sharpe_ratio=2.0, target_sharpe=2.0)
    assert s1 < s2 <= base


def test_volatility_cap():
    assert volatility_cap(0.10, max_fraction=0.05) == 0.05


def test_underdog_boost():
    boosted = underdog_boost(0.02, odds=3.0, threshold=2.5, boost=2.0)
    assert boosted > 0.02


def test_market_neutral():
    stake = market_neutral([0.1, -0.1, 0.2, -0.2])
    assert 0 <= stake <= 1