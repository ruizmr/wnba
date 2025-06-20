"""Smoke tests for :class:`python.data.multileague_pbp_datamodule.MultileaguePBPDataModule`."""

import itertools

import torch

from python.data.multileague_pbp_datamodule import MultileaguePBPDataModule


def test_dataloaders_iterate(tmp_path) -> None:  # type: ignore[missing-type-doc]
    """DataLoaders should yield batches without crashing."""
    dm = MultileaguePBPDataModule(data_dir=tmp_path, batch_size=16, num_workers=0)
    dm.prepare_data()
    dm.setup()

    loaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]

    for loader in loaders:
        batch = next(iter(loader))
        # Assert tensor shapes align with batch_size (except val/test could be smaller)
        assert "home_score" in batch
        assert isinstance(batch["home_score"], torch.Tensor)
        assert batch["home_score"].shape[0] <= 16


def test_split_sums_to_total() -> None:  # type: ignore[missing-type-doc]
    dm = MultileaguePBPDataModule(batch_size=64)
    dm.setup()

    total = sum(len(ds) for ds in (dm._train_ds, dm._val_ds, dm._test_ds))  # type: ignore[arg-type]
    expected = 200 * 3  # 3 leagues, default n_rows=200
    assert total == expected