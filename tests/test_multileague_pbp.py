"""Unit tests for the Multileague play-by-play schema and synthetic fixture generation."""

from python.data.schema import PbpEvent, generate_synthetic_pbp_events


def test_generate_counts_per_league() -> None:
    """Synthetic generator should yield exactly 200 rows per league."""
    for league in ("nba", "wnba", "ncaa_w"):
        events = generate_synthetic_pbp_events(league=league, n_rows=200, seed=123)
        assert len(events) == 200, league
        assert all(event.league == league for event in events), league


def test_schema_round_trip() -> None:
    """Serialising to dict and back should preserve field values."""
    sample_event = generate_synthetic_pbp_events(seed=1)[0]
    as_dict = sample_event.dict()
    loaded = PbpEvent.parse_obj(as_dict)
    assert loaded == sample_event


def test_scores_non_negative() -> None:
    """Scores must always be non-negative as the game progresses."""
    events = generate_synthetic_pbp_events(seed=42)
    for ev in events:
        assert ev.home_score >= 0
        assert ev.away_score >= 0