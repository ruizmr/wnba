"""Nightly data fetch job.

This script is intended to run via Agent 2's cron job wrapper. It pulls the latest
sportsbook lines and results (stubbed for now), converts them into Parquet
partitions, and writes them to `data/raw/{ds}/lines` and `data/raw/{ds}/results`.

The first iteration runs entirely on the local filesystem so that developers can
iterate quickly on laptops without S3 credentials. Migrating to a cloud object
store later will be trivial – simply replace the `output_path` string with an
`s3://…` URI supported by Ray Data.

Usage (local CPU)
-----------------
$ python -m python.data.nightly_fetch --date 2024-01-01 --rows 1000

When executed under Ray, set `RAY_ADDRESS` or `ray.init(address="auto")` to
connect to an existing cluster. Running *without* an address will start a local
Ray runtime automatically.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Callable, List

# ------------------------------------------------------------------
# Third-party dependencies with actionable error messages
# ------------------------------------------------------------------

try:
    import ray  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The `ray` library is required for nightly_fetch. Install it with\n"
        "    pip install \"ray[default,air,data]\"\n"
        "or recreate the supplied Conda environment (`conda env create -f env.yml`)."
    ) from exc

try:
    import ray.data  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "`ray.data` is missing. Make sure you installed Ray with the *data* extra (ray[data])."
    ) from exc

# -----------------------------------------------------------------------------
# Real‐data orchestration layer
# -----------------------------------------------------------------------------

# We delegate league-specific scraping to the dedicated fetchers that already
# live in this package.  `nightly_fetch.py` becomes a thin coordinator that
# calls them with a *date-scoped* output directory so historical partitions sit
# under `data/raw/<YYYY-MM-DD>/<league>/...`.

from python.data import (
    fetch_wnba_stats,
    fetch_nba_stats,
    fetch_ncaaw_stats,
)

# Mapping of league → (module, default seasons resolver)
_LEAGUES: dict[str, tuple[object, Callable[[date], List[str]]]] = {
    "wnba": (
        fetch_wnba_stats,
        lambda d: [str(d.year)],  # WNBA season == calendar year since summer season
    ),
    "nba": (
        fetch_nba_stats,
        lambda d: [f"{d.year-1}-{str(d.year)[-2:]}"] if d.month >= 8 else [
            f"{d.year-2}-{str(d.year-1)[-2:]}"
        ],  # e.g. 2025-06 → 2024-25 season
    ),
    "ncaaw": (
        fetch_ncaaw_stats,
        lambda d: [str(d.year - 1) if d.month < 7 else str(d.year)],  # academic year
    ),
}

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Entry point for the nightly fetch script."""

    parser = argparse.ArgumentParser(description="Nightly data fetch job")
    parser.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=date.today())
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Root output directory")
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    # Iterate over each league and fire its scraper.
    for league, (mod, season_fn) in _LEAGUES.items():
        seasons = season_fn(args.date)

        # Normalise to list of ints/strings accepted by each module.
        if not isinstance(seasons, list):
            seasons = [seasons]

        out_dir = (args.output_dir / args.date.isoformat() / league).resolve()
        print(f"\n🏀 [{league.upper()}] → seasons {seasons} → {out_dir}")

        # Each fetcher exposes a CLI-style main(argv) – we emulate that so we
        # don't spawn new Python processes.
        argv = ["--seasons", *seasons]
        argv += ["--out_dir", str(out_dir)]
        argv += ["--threads", "8"]

        try:
            mod.main(argv)  # type: ignore[attr-defined]
        except SystemExit:
            # The downstream scripts call argparse which raises SystemExit; we
            # swallow it so the loop continues.
            pass

    print(f"✅ Completed nightly fetch for {args.date.isoformat()}")


if __name__ == "__main__":
    main()