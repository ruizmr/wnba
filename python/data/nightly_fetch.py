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
from typing import List

try:
    import ray
    from ray.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Ray is required to run nightly_fetch. Install via `pip install ray[default]` or ensure it's in env.yml."
    ) from exc

from python.data.schema import LineRow, ResultRow

# -----------------------------------------------------------------------------
# Fake data generators – **placeholder** implementation until real API wired.
# -----------------------------------------------------------------------------

def _fake_lines(n: int, game_day: date) -> List[dict]:
    """Generate *n* fake line snapshots for a single day."""

    rows = []
    for i in range(n):
        game_id = int(game_day.strftime("%Y%m%d")) * 100 + i // 2
        rows.append(
            LineRow(
                game_id=game_id,
                team=["LAL", "NYK"][i % 2],
                line_type="spread",
                value=-3.5 if i % 2 == 0 else 3.5,
                odds=-110,
                timestamp=datetime.utcnow(),
            ).dict()
        )
    return rows


def _fake_results(num_games: int, game_day: date) -> List[dict]:
    """Generate fake final results for *num_games* games."""

    rows: List[dict] = []
    for i in range(num_games):
        game_id = int(game_day.strftime("%Y%m%d")) * 100 + i
        rows.extend(
            [
                ResultRow(
                    game_id=game_id,
                    team="LAL",
                    points=102 + i,
                    won=True,
                    timestamp=datetime.utcnow(),
                ).dict(),
                ResultRow(
                    game_id=game_id,
                    team="NYK",
                    points=97 + i,
                    won=False,
                    timestamp=datetime.utcnow(),
                ).dict(),
            ]
        )
    return rows

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Entry point for the nightly fetch script."""

    parser = argparse.ArgumentParser(description="Nightly data fetch job")
    parser.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=date.today())
    parser.add_argument("--rows", type=int, default=1000, help="Number of line snapshots to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Root output directory")
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    # Build datasets.
    ds_lines: Dataset = ray.data.from_items(_fake_lines(args.rows, args.date))
    ds_results: Dataset = ray.data.from_items(_fake_results(args.rows // 2, args.date))

    # Write Parquet partitions.
    ds_dir = args.output_dir / args.date.isoformat()
    ds_lines.write_parquet(str(ds_dir / "lines"), mode="overwrite")
    ds_results.write_parquet(str(ds_dir / "results"), mode="overwrite")

    print(f"✅ Wrote datasets to {ds_dir.resolve()}")


if __name__ == "__main__":
    main()