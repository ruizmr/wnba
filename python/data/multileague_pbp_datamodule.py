"""multileague_pbp_datamodule.py
Quick & simple PyTorch-Lightning–style DataModule that concatenates
Parquet play-by-play files from NBA, WNBA and NCAA-Women into one large
DataFrame then yields DataLoader of tensors ready for model pre-training.

It assumes each Parquet uses the unified `SCHEMA_COLS` defined in
fetch_ncaaw_stats.py and siblings.
Only minimal numeric engineering is performed here; heavy graph building is
Agent-1's job.  The module simply provides a per-event tensor of categorical
encoded columns + period/time features.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, random_split

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

NUMERIC_COLS = [
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
]

TIME_COLS = ["PCTIMESTRING"]  # convert mm:ss to remaining seconds


class PBPEventsDataset(Dataset):
    def __init__(self, parquet_paths: List[Path]):
        self.paths = parquet_paths
        self.offsets: List[int] = []  # cumulative row counts to allow __getitem__ mapping
        self.tables: List[pq.ParquetFile] = []
        cum = 0
        for p in self.paths:
            pf = pq.ParquetFile(p)
            n = pf.metadata.num_rows
            self.tables.append(pf)
            cum += n
            self.offsets.append(cum)

    def __len__(self):
        return self.offsets[-1] if self.offsets else 0

    def _idx_to_table(self, idx: int):
        # binary search offsets
        lo, hi = 0, len(self.offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < self.offsets[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        table_idx = lo
        prev_offset = self.offsets[table_idx - 1] if table_idx > 0 else 0
        row_idx = idx - prev_offset
        return table_idx, row_idx

    def __getitem__(self, idx: int):
        t_idx, r_idx = self._idx_to_table(idx)
        row = self.tables[t_idx].read_row_group(0, columns=SCHEMA_COLS).slice(r_idx, 1).to_pandas().iloc[0]
        # basic numeric tensor
        features = []
        for col in NUMERIC_COLS:
            features.append(row[col])
        # time remaining in seconds
        mm_ss = row["PCTIMESTRING"]
        try:
            m, s = mm_ss.split(":")
            rem_sec = int(m) * 60 + int(s)
        except Exception:
            rem_sec = 0
        features.append(rem_sec)
        return torch.tensor(features, dtype=torch.float32)


class MultiLeaguePBPDataModule:
    def __init__(
        self,
        data_roots: List[Path],
        batch_size: int = 512,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        self.data_roots = data_roots
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self):
        parquet_paths = []
        for root in self.data_roots:
            parquet_paths.extend([Path(p) for p in glob.glob(str(root / "**/*.parquet"), recursive=True)])
        dataset = PBPEventsDataset(parquet_paths)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.ds_train, self.ds_val = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )