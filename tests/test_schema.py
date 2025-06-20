from datetime import datetime

import pytest  # type: ignore

from python.data.schema import LineRow, ResultRow


def test_line_row_valid():
    now = datetime.utcnow()
    row = LineRow(
        game_id=123,
        team="LAL",
        line_type="spread",
        value=-3.5,
        odds=-110,
        timestamp=now,
    )
    assert row.game_id == 123
    assert row.team == "LAL"
    assert row.value == -3.5


def test_line_row_invalid_odds():
    with pytest.raises(ValueError):
        LineRow(
            game_id=1,
            team="NYK",
            line_type="total",
            value=220.5,
            odds=0,
            timestamp=datetime.utcnow(),
        )


def test_result_row():
    now = datetime.utcnow()
    res = ResultRow(
        game_id=123,
        team="LAL",
        points=102,
        won=True,
        timestamp=now,
    )
    assert res.points == 102
    assert res.won is True


def test_result_row_negative_points():
    with pytest.raises(ValueError):
        ResultRow(
            game_id=123,
            team="LAL",
            points=-5,
            won=False,
            timestamp=datetime.utcnow(),
        )