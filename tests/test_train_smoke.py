import os
import sys
from pathlib import Path
from importlib import import_module

import pytest

pytestmark = pytest.mark.skipif(
    not Path(os.getenv("WEHOOP_DATA_PATH", "/tmp/wehoop/wnba")).exists(),
    reason="WEHOOP dataset unavailable",
)


def test_training_smoke(monkeypatch):
    """Run the training entry-point in --dev-mode to ensure wiring works."""

    # Ensure Ray is initialised in local mode
    try:
        import ray  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("ray not installed")

    if not ray.is_initialized():
        ray.init(local_mode=True, num_cpus=2, namespace="train-smoke")

    # Build CLI argv equivalent
    argv = [
        "--seasons",
        "2025",
        "--num-samples",
        "1",
        "--epochs",
        "1",
        "--dev-mode",
    ]

    monkeypatch.setattr(sys, "argv", ["train.py", *argv])

    train_mod = import_module("python.model.train")
    # Call main; it will exit quickly in dev-mode
    train_mod.main(argv)

    ray.shutdown()