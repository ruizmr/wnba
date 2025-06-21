from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class LineRow(BaseModel):
    """Minimal event-level schema used during graph building.

    Note: This is a trimmed version that contains only the fields required by
    current training / serving pipelines.  Additional columns present in the
    raw WEHOOP play-by-play parquet files are intentionally ignored at this
    layer – they are either dropped during ingestion or surfaced through
    feature views.
    """

    game_id: int
    event_time: datetime = Field(alias="game_date_time")
    season: int

    home_team_id: int
    away_team_id: int

    period_number: Optional[int] = None
    clock_display_value: Optional[str] = None

    home_score: Optional[int] = None
    away_score: Optional[int] = None

    home_team_spread: Optional[float] = None
    game_spread: Optional[float] = None


class ResultRow(BaseModel):
    """Team-game level aggregate used as target during model training."""

    game_id: int
    season: int

    team_id: int
    opponent_team_id: int

    team_score: int
    opponent_team_score: int
    win_flag: bool


class ScheduleRow(BaseModel):
    """Game-level schedule metadata coming from WEHOOP schedules parquet."""

    game_id: int = Field(alias="id")
    season: int
    season_type: Optional[int] = None

    scheduled_start: datetime = Field(alias="game_date_time")
    game_date: date

    home_team_id: int = Field(alias="home_id")
    away_team_id: int = Field(alias="away_id")

    status: Optional[str] = Field(None, alias="status_type_state")


class PlayerBoxRow(BaseModel):
    """Player-game statistics row coming from WEHOOP player_box parquet."""

    game_id: int
    season: int

    athlete_id: int
    team_id: int

    minutes: Optional[float] = None
    points: Optional[int] = None
    rebounds: Optional[int] = None
    assists: Optional[int] = None
    plus_minus: Optional[float] = Field(None, alias="plus_minus")


__all__ = [
    "LineRow",
    "ResultRow",
    "ScheduleRow",
    "PlayerBoxRow",
]