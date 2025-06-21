import os
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pq = None  # type: ignore
    _PqTable = Any
else:
    from typing import Any as _Any
import pytest

if pq is None:
    pytest.skip("pyarrow not installed", allow_module_level=True)

# Import models only if pyarrow present to avoid unnecessary dependency chain.
from python.data.schema import (
    LineRow,
    ResultRow,
    ScheduleRow,
    PlayerBoxRow,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

WEHOOP_BASE = Path(os.getenv("WEHOOP_DATA_PATH", "/tmp/wehoop/wnba"))

# If pyarrow missing, mark entire module to be skipped.
pytestmark = pytest.mark.skipif(pq is None, reason="pyarrow not installed")


@pytest.fixture(scope="session")
def wehoop_available() -> bool:
    """Return True if the WEHOOP parquet repo is accessible locally."""
    return WEHOOP_BASE.exists()


def _first_row(folder_glob: str) -> dict:
    """Return the first record in the first parquet file that matches glob."""
    files = sorted(WEHOOP_BASE.glob(folder_glob))
    if not files:
        pytest.skip(f"No parquet files found for pattern: {folder_glob}")
    table = pq.read_table(files[0], columns=None, use_threads=False)  # type: ignore
    return {col: table[col][0].as_py() for col in table.schema.names}


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_line_row_parsing(wehoop_available):
    if not wehoop_available:
        pytest.skip("WEHOOP data repo not available; set WEHOOP_DATA_PATH to run")
    record = _first_row("pbp/parquet/*.parquet")
    model = LineRow.parse_obj(record)
    assert model.game_id is not None
    assert model.season >= 2000


def test_result_row_parsing(wehoop_available):
    if not wehoop_available:
        pytest.skip("WEHOOP data repo not available; set WEHOOP_DATA_PATH to run")
    record = _first_row("team_box/parquet/*.parquet")
    model = ResultRow.parse_obj(record)
    assert model.team_id is not None
    assert model.team_score >= 0


def test_schedule_row_parsing(wehoop_available):
    if not wehoop_available:
        pytest.skip("WEHOOP data repo not available; set WEHOOP_DATA_PATH to run")
    record = _first_row("schedules/parquet/*.parquet")
    model = ScheduleRow.parse_obj(record)
    assert model.game_id is not None
    assert model.scheduled_start is not None


def test_player_box_row_parsing(wehoop_available):
    if not wehoop_available:
        pytest.skip("WEHOOP data repo not available; set WEHOOP_DATA_PATH to run")
    record = _first_row("player_box/parquet/*.parquet")
    model = PlayerBoxRow.parse_obj(record)
    assert model.athlete_id is not None
    assert model.team_id is not None