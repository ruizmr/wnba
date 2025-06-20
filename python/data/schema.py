"""Schema definitions for play-by-play (PBP) events across multiple basketball leagues.

Example
-------
>>> from python.data.schema import generate_synthetic_pbp_events
>>> events = generate_synthetic_pbp_events(league="nba", n_rows=10, seed=42)
>>> events[0]
PbpEvent(game_id='nba-2024-1', event_id=1, league='nba', ...)
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Literal, Sequence

from pydantic import BaseModel, Field, validator

__all__ = [
    "PbpEvent",
    "generate_synthetic_pbp_events",
]

LeagueLiteral = Literal["nba", "wnba", "ncaa_w"]

EVENT_TYPES: Sequence[str] = (
    "shot",
    "free_throw",
    "rebound",
    "turnover",
    "foul",
    "sub",
    "timeout",
)


class PbpEvent(BaseModel):
    """Unified play-by-play event schema supporting NBA, WNBA and NCAA-W."""

    game_id: str = Field(
        ..., description="Composite id: <league>-<season>-<uuid|int>", example="nba-2024-42"
    )
    event_id: int = Field(..., ge=0, description="Monotonically increasing id within a game")
    league: LeagueLiteral
    season: int = Field(..., ge=2000, description="Year that season started, e.g., 2024")
    period: int = Field(..., ge=1, le=20)
    clock: float = Field(..., ge=0.0, description="Seconds remaining in the period")
    offense_team_id: str
    defense_team_id: str
    player_id: str | None = None
    event_type: Literal[
        "shot",
        "free_throw",
        "rebound",
        "turnover",
        "foul",
        "sub",
        "timeout",
    ]
    points: int = Field(..., ge=0, le=3)
    home_score: int = Field(..., ge=0)
    away_score: int = Field(..., ge=0)
    vegas_line: float | None = Field(
        None, description="Closing Vegas line at tip-off (home – away spread)"
    )
    timestamp_utc: datetime

    # ----------------------- Validators -----------------------

    @validator("home_score", "away_score")
    def _non_negative(cls, v: int) -> int:  # noqa: D401, N805
        if v < 0:
            raise ValueError("Scores must be non-negative")
        return v

    class Config:  # noqa: D401
        """Model configuration."""

        extra = "forbid"
        validate_assignment = True
        allow_mutation = False


# ---------------------------------------------------------------------------
# Synthetic data generation helpers (for unit tests & examples only)
# ---------------------------------------------------------------------------

def _random_game_id(league: LeagueLiteral, season: int, game_idx: int) -> str:
    return f"{league}-{season}-{game_idx}"


def generate_synthetic_pbp_events(
    *,
    league: LeagueLiteral = "nba",
    season: int = 2024,
    n_rows: int = 200,
    seed: int | None = None,
) -> list[PbpEvent]:
    """Generate *n_rows* synthetic :class:`PbpEvent` objects.

    The generator is **deterministic** for a fixed ``seed``.  It creates a single
    fictitious game with incrementing *event_id*s and plausible scores.
    """

    rng = random.Random(seed)

    # Simple game clock model: 4 periods × 720 seconds
    period_durations = {1: 720, 2: 720, 3: 720, 4: 720}

    events: list[PbpEvent] = []
    home_score = 0
    away_score = 0

    game_id = _random_game_id(league, season, 1)
    start_ts = datetime.utcnow().replace(microsecond=0)

    offense_team_id = f"{league}_home"
    defense_team_id = f"{league}_away"

    for i in range(1, n_rows + 1):
        period = rng.choice([1, 2, 3, 4])
        clock_remaining = rng.uniform(0, period_durations[period])
        event_type = rng.choice(EVENT_TYPES)

        # Points scored only on shots or free throws
        if event_type == "shot":
            points = rng.choice([0, 2, 3])
        elif event_type == "free_throw":
            points = rng.choice([0, 1])
        else:
            points = 0

        # Randomly decide if home or away scored
        if points > 0:
            if rng.random() < 0.5:
                home_score += points
            else:
                away_score += points

        timestamp_utc = start_ts + timedelta(seconds=i * 5)

        event = PbpEvent(
            game_id=game_id,
            event_id=i,
            league=league,
            season=season,
            period=period,
            clock=clock_remaining,
            offense_team_id=offense_team_id,
            defense_team_id=defense_team_id,
            player_id=str(uuid.uuid4()) if rng.random() < 0.9 else None,
            event_type=event_type,  # type: ignore[arg-type]
            points=points,
            home_score=home_score,
            away_score=away_score,
            vegas_line=rng.uniform(-12.5, 12.5) if i == 1 else None,
            timestamp_utc=timestamp_utc,
        )
        events.append(event)

    return events