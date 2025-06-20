"""fetch_wnba_stats.py
---------------------------------
Pulls WNBA play-by-play data from stats.wnba.com for multiple seasons and writes
partitioned Parquet files to `data/wnba/raw_playbyplay/SEASON/`.

Rationale
==========
High-quality historical play-by-play (PBP) is the best pre-training fuel for
Agent-1's heterogeneous graph builder. By automating an end-to-end fetch ➜ clean
➜ parquet pipeline we (a) guarantee reproducible datasets, (b) make local/
cloud training I/O efficient, and (c) open the door for daily incremental
updates once the season starts.

The script is intentionally lightweight (pure requests + pandas) to avoid heavy
dependencies.  It respects NBA Stats rate-limits (sleep between calls) and can
resume partially downloaded seasons.

Usage (CLI)
===========
Fetch regular-season PBP for 2020-2024:
    $ python -m python.data.fetch_wnba_stats --seasons 2020 2021 2022 2023 2024

Optional flags:
    --out_dir   Target directory (default: data/wnba/raw_playbyplay)
    --threads   Parallel fetch workers (default: 4)
    --force     Re-download and overwrite existing Parquet partitions.

Environment
===========
Requires pandas>=1.3, pyarrow, requests-cache (optional but recommended).
"stats.wnba.com" sometimes blocks generic user-agents – we spoof a common
browser UA.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests

# --- Constants ---------------------------------------------------------------
LEAGUE_ID = "10"  # WNBA league ID used by stats endpoints
BASE = "https://stats.wnba.com/stats"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://stats.wnba.com/",
}
RATE_LIMIT_SEC = 0.8  # conservative delay between hits

# -----------------------------------------------------------------------------

def get_games_for_season(season: int) -> pd.DataFrame:
    """Return DataFrame with columns GAME_ID, GAME_DATE, MATCHUP for season."""
    # leaguegamelog delivers one row per team; we deduplicate on GAME_ID.
    params = {
        "Counter": "0",
        "Direction": "DESC",
        "LeagueID": LEAGUE_ID,
        "PlayerOrTeam": "T",
        "Season": f"{season}",
        "SeasonType": "Regular+Season",
        "Sorter": "GAME_DATE",
    }
    url = f"{BASE}/leaguegamelog"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    headers = raw["resultSets"][0]["headers"]
    rows = raw["resultSets"][0]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    games = df.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE", "MATCHUP"]]
    return games


def fetch_pbp(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play events for a single game as DataFrame."""
    params = {
        "GameID": game_id,
        "StartPeriod": "0",
        "EndPeriod": "10",
    }
    url = f"{BASE}/playbyplayv2"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    headers = raw["resultSets"][0]["headers"]
    rows = raw["resultSets"][0]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "GAME_ID", game_id)
    return df


def save_parquet(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def process_game(game_id: str, season_dir: Path, force: bool = False):
    out_file = season_dir / f"{game_id}.parquet"
    if out_file.exists() and not force:
        return
    try:
        df = fetch_pbp(game_id)
        save_parquet(df, out_file)
        time.sleep(RATE_LIMIT_SEC)
    except Exception as e:
        print(f"Failed game {game_id}: {e}")


# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Fetch WNBA play-by-play data")
    parser.add_argument("--seasons", type=int, nargs="*", required=True, help="Season years, e.g. 2020 2021")
    parser.add_argument("--out_dir", type=Path, default=Path("data/wnba/raw_playbyplay"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-download even if parquet exists")
    args = parser.parse_args(argv)

    for season in args.seasons:
        print(f"=== Season {season} ===")
        season_dir = args.out_dir / str(season)
        games_df = get_games_for_season(season)
        game_ids = games_df["GAME_ID"].tolist()
        print(f"{len(game_ids)} games found")
        with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
            for gid in game_ids:
                ex.submit(process_game, gid, season_dir, args.force)

    print("Done.")


if __name__ == "__main__":
    main()