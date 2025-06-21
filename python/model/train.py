"""Mini training pipeline entry-point.

This is a *stub* implementation suitable for CI smoke-tests.  It demonstrates
how to wire the new `load_wehoop` ingestion helper into the graph builder and
accepts CLI flags for season selection and optional exclusion of pre-season
matches.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import List, Sequence

import ray
import time

from python.data.wehoop_ingest import load_wehoop
from python.graph.builder import build_graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_seasons(seasons: Sequence[str] | None) -> List[int] | None:
    if not seasons:
        return None
    return [int(s) for s in seasons]


def _filter_preseason(ds_schedules, exclude: bool):  # noqa: ANN001 – schedules dataset TBD
    if not exclude:
        return ds_schedules
    # season_type: 2 = regular season for ESPN feeds, 1 = preseason
    return ds_schedules.filter(lambda r: r.get("season_type", 2) == 2)


# ---------------------------------------------------------------------------
# Main training routine (placeholder)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="MiniHGT training pipeline")
    parser.add_argument("--seasons", nargs="*", help="Season years to include, e.g. 2023 2024")
    parser.add_argument("--exclude-preseason", action="store_true", help="Drop preseason games via schedules")
    parser.add_argument("--num-samples", type=int, default=2, help="Ray Tune – number of samples (stub)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (stub)")
    parser.add_argument("--dev-mode", action="store_true", help="Fast development mode (smoke CI)")
    args = parser.parse_args(argv)

    seasons = _parse_seasons(args.seasons)
    logger.info("Loading WEHOOP seasons=%s", seasons or "ALL")

    ingest_start = time.time()
    ds_lines, ds_results = load_wehoop(seasons=seasons)
    ingest_elapsed = time.time() - ingest_start
    total_rows = ds_lines.count() + ds_results.count()
    rows_per_sec = total_rows / ingest_elapsed if ingest_elapsed else 0

    logger.info("Ingest throughput: %.1f rows/s (Grafana metric)", rows_per_sec)

    # ------------------------------------------------------------------
    # Train/Val deterministic split (Season % 5 == 0)
    # ------------------------------------------------------------------
    train_lines = ds_lines.filter(lambda r: r["season"] % 5 != 0)
    val_lines = ds_lines.filter(lambda r: r["season"] % 5 == 0)

    train_results = ds_results.filter(lambda r: r["season"] % 5 != 0)
    val_results = ds_results.filter(lambda r: r["season"] % 5 == 0)

    logger.info(
        "Split rows – train_lines=%d val_lines=%d train_results=%d val_results=%d",
        train_lines.count(),
        val_lines.count(),
        train_results.count(),
        val_results.count(),
    )

    if args.exclude_preseason:
        logger.info("Excluding preseason games (season_type!=2)…")
        # Not implemented schedules dataset yet – placeholder

    # Build streaming graph (memory safe)
    stats = build_graph(ds_lines, ds_results, batch_size=2048 if args.dev_mode else 8192)

    logger.info(
        "Graph stats: %s teams=%d games=%d edges=%d",
        stats.seasons,
        stats.team_count,
        stats.game_count,
        stats.edge_count,
    )

    # Placeholder for actual training – simulate duration
    if args.dev_mode:
        logger.info("Dev-mode enabled, skipping heavy training loop.")
        return

    logger.info("Starting dummy training loop…")

    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d …", epoch + 1, args.epochs)
        time.sleep(1)

    logger.info("Training complete at %s", datetime.utcnow().isoformat())


if __name__ == "__main__":
    main()