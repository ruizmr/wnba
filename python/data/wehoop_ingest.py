"""Streaming ingestion helpers that convert raw WEHOOP parquet files into
Ray `Dataset`s that conform to our internal schemas (LineRow & ResultRow).

The ingestion is purposely *lazy* – nothing is loaded until the caller starts
consuming / transforming the returned Ray Datasets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import shutil

try:
    import ray
except ImportError as exc:  # pragma: no cover – ray may be optional in CI
    raise RuntimeError(
        "`ray` is required for wehoop_ingest; install with `pip install ray`."
    ) from exc

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Where the WEHOOP parquet repo lives locally.  You can override via env-var if
# you keep the dataset elsewhere.
WEHOOP_ROOT = Path(os.getenv("WEHOOP_DATA_PATH", "/tmp/wehoop/wnba"))

# Parquet sub-folders & file naming patterns used by the upstream repo.
_PBP_PATH = "pbp/parquet"
_PBP_PATTERN = "play_by_play_{year}.parquet"

_TEAM_BOX_PATH = "team_box/parquet"
_TEAM_BOX_PATTERN = "team_box_{year}.parquet"


def _discover_parquet_files(
    base: Path,
    sub_path: str,
    pattern: str,
    seasons: Iterable[int] | None,
) -> List[str]:
    """Return absolute file paths for the requested seasons.

    If *seasons* is *None* we will simply glob for every file that matches the
    pattern in *sub_path*.
    """

    folder = base / sub_path
    if not folder.exists():
        raise FileNotFoundError(f"WEHOOP directory not found: {folder}")

    paths: list[str] = []
    if seasons is None:
        # Wild-card glob (pattern may include the {year} placeholder)
        paths = [str(p) for p in folder.glob(pattern.format(year="*"))]
    else:
        for year in seasons:
            p = folder / pattern.format(year=year)
            if p.exists():
                paths.append(str(p))
            else:
                logger.warning("Missing WEHOOP parquet for %s – skipping", p)
    if not paths:
        raise FileNotFoundError(
            f"No parquet files found for seasons {seasons} in {folder}"
        )
    return paths


# ----------------------------------------------------------------------------
# Column projection & cleaning helpers
# ----------------------------------------------------------------------------

# Minimal column sets needed for the Pydantic models.  We drop everything else
# during the `map_batches` so we don't carry around dozens of unused columns.
_LINES_COLUMNS = [
    "game_id",
    "game_date_time",
    "season",
    "home_team_id",
    "away_team_id",
    "period_number",
    "clock_display_value",
    "home_score",
    "away_score",
    "home_team_spread",
    "game_spread",
]

_RESULTS_COLUMNS = [
    "game_id",
    "season",
    "team_id",
    "opponent_team_id",
    "team_score",
    "opponent_team_score",
    "team_winner",
]


def _project_lines(batch: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-defined]
    """Return dataframe restricted to the columns we actually use."""

    batch = batch[_LINES_COLUMNS].copy()  # type: ignore[index]
    # Replace empty strings with None – helps numeric conversion later.
    return batch.replace({"": None})  # type: ignore[call-arg]


def _project_results(batch: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-defined]
    df = batch[_RESULTS_COLUMNS].copy()  # type: ignore[index]
    # Rename to internal field names (only where different)
    df = df.rename(columns={"team_winner": "win_flag"})  # type: ignore[arg-type]
    return df


# ---------------------------------------------------------------------------
# Optional caching (15 GB threshold)
# ---------------------------------------------------------------------------

_CACHE_THRESHOLD_BYTES = 15 * 1024 ** 3  # 15 GB
_CACHE_ROOT = (
    Path(os.getenv("DATA_CACHE", "/tmp/data_cache")) / "wehoop"
)


def _maybe_cache(paths: List[str]) -> List[str]:
    """Stage parquet files to a local cache directory if data volume is large.

    Logic: If the combined size of *paths* exceeds the 15 GB threshold, copy
    the files into `$DATA_CACHE/wehoop/…` (maintaining the same relative
    folder structure).  On subsequent runs, the already-cached files will be
    used directly, avoiding heavy re-reads from remote/network filesystems.
    """

    total_size = 0
    try:
        total_size = sum(Path(p).stat().st_size for p in paths)
    except FileNotFoundError:
        # If any file is missing just bail – discovery layer will raise later.
        return paths

    if total_size <= _CACHE_THRESHOLD_BYTES:
        return paths  # not big enough – skip caching

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    cached: list[str] = []
    for p in paths:
        src = Path(p)
        try:
            rel = src.relative_to(WEHOOP_ROOT)
        except ValueError:
            rel = src.name  # fallback – put flat in cache
        dest = _CACHE_ROOT / rel
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Caching WEHOOP parquet -> %s", dest)
            shutil.copy2(src, dest)
        cached.append(str(dest))

    return cached


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def load_wehoop(seasons: list[int] | None = None) -> Tuple["ray.data.Dataset", "ray.data.Dataset"]:  # noqa: D401,E501
    """Return **(ds_lines, ds_results)** in internal schemas.

    Parameters
    ----------
    seasons
        Explicit list of seasons (years) – e.g. ``[2023, 2024]``.  If *None*
        we load every season available in the local WEHOOP repo.
    """

    if not WEHOOP_ROOT.exists():
        raise FileNotFoundError(
            "WEHOOP repo not found – clone it and/or set WEHOOP_DATA_PATH."
        )

    # Discover parquet paths
    lines_paths = _discover_parquet_files(
        WEHOOP_ROOT, _PBP_PATH, _PBP_PATTERN, seasons
    )
    results_paths = _discover_parquet_files(
        WEHOOP_ROOT, _TEAM_BOX_PATH, _TEAM_BOX_PATTERN, seasons
    )

    # Optionally stage to cache if dataset is large
    lines_paths = _maybe_cache(lines_paths)
    results_paths = _maybe_cache(results_paths)

    # ---------------------------------------------------------------------
    # Read via Ray
    # ---------------------------------------------------------------------
    logger.info("Reading %d play-by-play parquet files", len(lines_paths))
    ds_lines = ray.data.read_parquet(lines_paths)

    logger.info("Reading %d team-box parquet files", len(results_paths))
    ds_results = ray.data.read_parquet(results_paths)

    # ---------------------------------------------------------------------
    # Projection & cleaning (streaming, preserves lazy eval)
    # ---------------------------------------------------------------------
    ds_lines = ds_lines.map_batches(_project_lines, batch_format="pandas")

    ds_results = (
        ds_results.map_batches(_project_results, batch_format="pandas")
        # quality filter – drop obvious corrupt rows
        .filter(lambda r: r["team_id"] is not None and r["opponent_team_id"] is not None)
    )

    return ds_lines, ds_results