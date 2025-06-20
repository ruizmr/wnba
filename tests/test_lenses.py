import math

import pytest

from python.lenses import (
    decimal_to_implied_prob,
    edge,
    expected_value,
    kelly_fraction,
)


@pytest.mark.parametrize(
    "decimal_odds,expected",
    [
        (2.0, 0.5),
        (1.91, pytest.approx(1 / 1.91)),
        (3.5, pytest.approx(1 / 3.5)),
    ],
)
def test_decimal_to_implied_prob(decimal_odds, expected):
    assert decimal_to_implied_prob(decimal_odds) == expected


@pytest.mark.parametrize(
    "win_prob,decimal_odds,expected",
    [
        # fair bet (zero EV)
        (0.5, 2.0, 0.0),
        # positive EV
        (0.55, 2.0, 0.55 * 1 - 0.45),
        # negative EV
        (0.45, 2.0, 0.45 * 1 - 0.55),
    ],
)
def test_expected_value(win_prob, decimal_odds, expected):
    assert expected_value(win_prob, decimal_odds) == pytest.approx(expected)


@pytest.mark.parametrize(
    "win_prob,decimal_odds,expected",
    [
        (0.55, 2.0, 0.05),  # 55% vs implied 50%
        (0.5, 2.0, 0.0),
        (0.25, 4.0, -0.0),
    ],
)
def test_edge(win_prob, decimal_odds, expected):
    assert edge(win_prob, decimal_odds) == pytest.approx(expected)


@pytest.mark.parametrize(
    "win_prob,decimal_odds,max_fraction,expected",
    [
        # textbook example: 60% coin, even odds → 20% of bankroll
        (0.6, 2.0, 1.0, 0.2),
        # if negative EV, stake should be 0
        (0.4, 2.0, 1.0, 0.0),
        # Half Kelly via multiplier
        (0.6, 2.0, 0.5, 0.1),
        # implied > subjective: no bet
        (0.05, 5.0, 1.0, 0.0),
    ],
)
def test_kelly_fraction_basic(win_prob, decimal_odds, max_fraction, expected):
    assert kelly_fraction(win_prob, decimal_odds, multiplier=max_fraction) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("win_prob", [-0.1, 1.1])
def test_kelly_fraction_invalid_prob(win_prob):
    with pytest.raises(ValueError):
        kelly_fraction(win_prob, 2.0)


def test_kelly_fraction_invalid_odds():
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 1.0)


def test_decimal_odds_invalid():
    with pytest.raises(ValueError):
        decimal_to_implied_prob(1.0)