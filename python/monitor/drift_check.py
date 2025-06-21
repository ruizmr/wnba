"""Daily schedule row-count drift check.

Intended to be executed via cron (see scripts/cron_drift.sh).  Requires
WEHOOP_DATA_PATH env-var to point at the local clone of the WEHOOP repo and a
SLACK_WEBHOOK_URL variable for notifications.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pyarrow.parquet as pq
import requests

WEHOOP_ROOT = Path(os.getenv("WEHOOP_DATA_PATH", "/tmp/wehoop/wnba"))
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_schedule_rows(target_date: date) -> int:
    """Return count of schedule rows for *target_date* across all seasons."""

    counts = 0
    for parquet in (WEHOOP_ROOT / "schedules" / "parquet").glob("*.parquet"):
        tbl = pq.read_table(parquet, columns=["game_date"])
        mask = tbl.column(0) == target_date.isoformat()
        counts += mask.sum().as_py()
    return counts


def _post_slack(msg: str) -> None:
    if not SLACK_WEBHOOK:
        logger.warning("Slack webhook not configured – message: %s", msg)
        return
    requests.post(SLACK_WEBHOOK, data=json.dumps({"text": msg}), timeout=5)


def main() -> None:  # noqa: D401
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Build history for 7 days prior to yesterday (excludes sample date)
    history_dates: List[date] = [yesterday - timedelta(days=i) for i in range(1, 8)]

    y_count = _load_schedule_rows(yesterday)
    hist_counts = [_load_schedule_rows(d) for d in history_dates]
    median_hist = statistics.median(hist_counts) if hist_counts else 0

    if median_hist == 0:
        logger.warning("Cannot compute drift – median history is zero.")
        return

    delta_pct = abs(y_count - median_hist) / median_hist * 100
    logger.info("Schedule rows yesterday=%d median=%d Δ=%.1f%%", y_count, median_hist, delta_pct)

    if delta_pct > 25:
        _post_slack(
            f"⚠️ WEHOOP data drift: yesterday had {y_count} schedule rows vs median {median_hist} (Δ {delta_pct:.1f}%)."
        )


if __name__ == "__main__":
    main()