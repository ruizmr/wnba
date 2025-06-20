"""fetch_ncaaw_stats.py
Download NCAA Women play-by-play CSV dumps published by
`https://github.com/sportsdataverse/ncaaw-fastR-data` (fastR repository).
Each season lives in a single compressed CSV at:
    https://raw.githubusercontent.com/sportsdataverse/ncaaw-fastR-data/master/play_by_play/ncaaw_pbp_{season}.csv.gz
where season is the starting year (e.g. 2020 for 2020-21).

We convert each giant CSV into individual Parquet files per GAME_ID under
`data/ncaaw/raw_playbyplay/<season>/GAME_ID.parquet` using the *same* column
subset & names used by the NBA/WNBA fetchers so that downstream graph code can
concatenate seamlessly.
"""
from __future__ import annotations

import argparse
import gzip
import io
import textwrap
from pathlib import Path
from typing import List

import pandas as pd
import requests

SCHEMA_COLS = [
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "WCTIMESTRING",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
]

BASE_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/ncaaw-fastR-data/master/"
    "play_by_play/ncaaw_pbp_{season}.csv.gz"
)


NCAA_TO_NBA_MSGTYPE = {
    # fastR's event_type -> NBA EVENTMSGTYPE mapping (best-effort)
    "shot": 1,
    "miss": 2,
    "free throw": 3,
    "rebound": 4,
    "turnover": 5,
    "foul": 6,
    "violation": 7,
    "substitution": 8,
    "timeout": 9,
    "start of period": 12,
    "end of period": 13,
}


def download_season_csv(season: int) -> pd.DataFrame:
    url = BASE_URL.format(season=season)
    print(f"Downloading {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    buf = gzip.decompress(r.content)
    df = pd.read_csv(io.BytesIO(buf))
    return df


def convert_schema(df: pd.DataFrame) -> pd.DataFrame:
    # fastR naming refs: https://github.com/sportsdataverse/ncaaw-fastR-data
    df_out = pd.DataFrame()
    df_out["GAME_ID"] = df["game_id"].astype(str)
    df_out["EVENTNUM"] = df.groupby("game_id").cumcount() + 1
    df_out["EVENTMSGTYPE"] = df["play_type"].map(NCAA_TO_NBA_MSGTYPE).fillna(0).astype(int)
    df_out["EVENTMSGACTIONTYPE"] = 0  # unavailable
    df_out["PERIOD"] = df["period"]
    df_out["WCTIMESTRING"] = df["half_clock"].fillna("")
    df_out["PCTIMESTRING"] = df["period_seconds_remaining"].fillna("")
    df_out["HOMEDESCRIPTION"] = df["home_desc"].fillna("")
    df_out["NEUTRALDESCRIPTION"] = df["description"].fillna("")
    df_out["VISITORDESCRIPTION"] = df["away_desc"].fillna("")
    df_out["SCORE"] = df["score"].fillna("")
    df_out["SCOREMARGIN"] = df["score_margin"].fillna("")
    return df_out[SCHEMA_COLS]


def season_pipeline(season: int, out_dir: Path, force: bool):
    raw = download_season_csv(season)
    raw = convert_schema(raw)
    for gid, gdf in raw.groupby("GAME_ID"):
        path = out_dir / str(season) / f"{gid}.parquet"
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(path, index=False)


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Fetch NCAA Women's play-by-play and convert to Parquet"
    )
    parser.add_argument("--seasons", nargs="*", type=int, required=True, help="Start years e.g. 2020 2021")
    parser.add_argument("--out_dir", type=Path, default=Path("data/ncaaw/raw_playbyplay"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    for season in args.seasons:
        season_pipeline(season, args.out_dir, args.force)
    print("Done.")


if __name__ == "__main__":
    main()