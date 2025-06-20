"""Tests for aggregation helpers."""
from python.aggregate import geo_mean


def test_geo_mean_identity():
    assert geo_mean([0.2]) == 0.2


def test_geo_mean_multiple():
    result = geo_mean([0.1, 0.4])
    assert 0.1 <= result <= 0.4


def test_geo_mean_empty_returns_zero():
    assert geo_mean([]) == 0.0