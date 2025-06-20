"""fetch_nba_stats.py
Fetch NBA play-by-play for given seasons and store Parquet files under
`data/nba/raw_playbyplay/<season>/`.

The code mirrors `fetch_wnba_stats.py` but uses LeagueID=00.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://stats.nba.com/",
}
BASE = "https://stats.nba.com/stats"
LEAGUE_ID = "00"
RATE = 0.6


def games_for_season(season: str) -> pd.DataFrame:
    params = {
        "Counter": "0",
        "Direction": "DESC",
        "LeagueID": LEAGUE_ID,
        "PlayerOrTeam": "T",
        "Season": season,
        "SeasonType": "Regular+Season",
        "Sorter": "GAME_DATE",
    }
    r = requests.get(BASE + "/leaguegamelog", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    headers = js["resultSets"][0]["headers"]
    rows = js["resultSets"][0]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    return df.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE", "MATCHUP"]]


def pbp(game_id: str) -> pd.DataFrame:
    params = {"GameID": game_id, "StartPeriod": "0", "EndPeriod": "10"}
    r = requests.get(BASE + "/playbyplayv2", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    headers = js["resultSets"][0]["headers"]
    rows = js["resultSets"][0]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "GAME_ID", game_id)
    return df


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def worker(gid: str, season_dir: Path, force: bool):
    out = season_dir / f"{gid}.parquet"
    if out.exists() and not force:
        return
    try:
        df = pbp(gid)
        save(df, out)
    except Exception as e:
        print("fail", gid, e)
    time.sleep(RATE)


def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", required=True, help="e.g. 2020-21 2021-22")
    ap.add_argument("--out_dir", type=Path, default=Path("data/nba/raw_playbyplay"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    for season in args.seasons:
        sd = args.out_dir / season
        games = games_for_season(season)
        print(season, len(games))
        with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
            for gid in games.GAME_ID.tolist():
                ex.submit(worker, gid, sd, args.force)
    print("done")


if __name__ == "__main__":
    main()