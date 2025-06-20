from __future__ import annotations

from datetime import datetime
from typing import Literal

# NOTE: We lazily import Pydantic to provide a helpful message if it's missing in the env.
try:
    from pydantic import BaseModel, Field, validator  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Pydantic is required for schema definitions. Run `pip install pydantic` or add it to env.yml."
    ) from exc

__all__ = ["LineRow", "ResultRow"]


class LineRow(BaseModel):
    """Single sportsbook line snapshot.

    Example
    -------
    >>> LineRow(
    ...     game_id=2024010101,
    ...     team="LAL",
    ...     line_type="spread",
    ...     value=-5.5,
    ...     odds=-110,
    ...     timestamp=datetime.utcnow(),
    ... )
    LineRow(game_id=2024010101, team='LAL', line_type='spread', value=-5.5, odds=-110, timestamp=datetime(...))
    """

    game_id: int = Field(..., description="Unique identifier for the basketball game.")
    team: str = Field(..., description="Three-letter team code, e.g. LAL for Los Angeles Lakers.")
    line_type: Literal["spread", "total", "moneyline"] = Field(..., description="Type of betting line.")
    value: float = Field(..., description="Line value, e.g. -5.5 points spread or 224.0 total points.")
    odds: int = Field(..., description="American odds, e.g. -110 or +220.")
    timestamp: datetime = Field(..., description="UTC timestamp when the line was captured.")

    # --- Validators -----------------------------------------------------------------

    @validator("odds")
    def _odds_not_zero(cls, v: int) -> int:  # noqa: N805, D401
        """Ensure that odds are non-zero."""
        if v == 0:
            raise ValueError("odds cannot be 0 – use American odds like -110 or +100")
        return v


class ResultRow(BaseModel):
    """Final box-score style result row.

    Example
    -------
    >>> ResultRow(
    ...     game_id=2024010101,
    ...     team="LAL",
    ...     points=102,
    ...     won=True,
    ...     timestamp=datetime.utcnow(),
    ... )
    ResultRow(game_id=2024010101, team='LAL', points=102, won=True, timestamp=datetime(...))
    """

    game_id: int = Field(..., description="Unique identifier for the basketball game.")
    team: str = Field(..., description="Three-letter team code.")
    points: int = Field(..., ge=0, description="Points scored by the team.")
    won: bool = Field(..., description="Whether this team won the game.")
    timestamp: datetime = Field(..., description="UTC timestamp when the result was recorded.")

    # --- Validators -----------------------------------------------------------------

    @validator("points")
    def _points_non_negative(cls, v: int) -> int:  # noqa: N805, D401
        """Points must be non-negative."""
        if v < 0:
            raise ValueError("points must be >= 0")
        return v