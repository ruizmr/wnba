"""Integration tests for Parquet partitioning of MultileaguePBPDataModule."""

import os
from pathlib import Path

import pytest

from python.data.multileague_pbp_datamodule import MultileaguePBPDataModule

# ---------------------------------------------------------------------------
# Skip entire module if pyarrow not available (CI will install it).
# ---------------------------------------------------------------------------

a = pytest.importorskip("pyarrow", reason="pyarrow not installed; install pyarrow to run parquet tests")
import pyarrow as pa  # type: ignore  # noqa: E402  pylint: disable=wrong-import-order
import pyarrow.dataset as ds  # type: ignore  # noqa: E402  pylint: disable=wrong-import-order


@pytest.mark.parametrize("league", ["nba", "wnba", "ncaa_w"])
def test_partition_paths_created(tmp_path: Path, league: str) -> None:
    """`prepare_data` should write at least one .parquet file under league/season partition."""
    dm = MultileaguePBPDataModule(data_dir=tmp_path, leagues=[league])
    dm.prepare_data()

    partition_dir = (
        tmp_path
        / f"league={league}"
        / "season=2024"  # default season in DM
    )
    parquet_files = list(partition_dir.glob("*.parquet"))
    assert parquet_files, f"no parquet files for {league}"  # at least one file exists


@pytest.mark.parametrize("league", ["nba", "wnba", "ncaa_w"])
def test_read_back_counts(tmp_path: Path, league: str) -> None:
    """Reading dataset via Arrow should return exactly 200 rows for the given league."""
    dm = MultileaguePBPDataModule(data_dir=tmp_path, leagues=[league])
    dm.prepare_data()

    # Read dataset via Arrow partition discovery
    dataset = ds.dataset(tmp_path, format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("league") == league)

    assert table.num_rows == 200, f"unexpected row count for {league}"
    # Check that all seasons equal 2024 in this test scenario
    seasons = set(table.column("season").to_pylist())
    assert seasons == {2024}