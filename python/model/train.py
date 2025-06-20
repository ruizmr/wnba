"""Training entrypoint for the `MiniHGT` model.

This script is launched by Ray **jobs** (`ray job submit`) or Ray **CLI**
(`ray submit`) and adapts automatically to CPU-only laptops *and* RunPod GPU
clusters.

Quick-start 🖥️  (local CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~
Create/activate the Conda env and run one smoke-test epoch::

    conda activate edge-engine
    ray job submit --runtime-env="{}" \
        python python/model/train.py --smoke-test --epochs 1

The command starts a *local* Ray runtime, runs a single hyper-parameter sample,
and exits in < 1 minute.

Production 🚀  (RunPod GPU cluster)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assuming Agent 2's cluster config is in `.ray/cluster.yaml` and the repo has
been pushed to the head node::

    ray submit .ray/cluster.yaml \
        python/model/train.py \
        --num-samples 8 --epochs 3

Ray Tune will orchestrate 8 trials across the A5000 GPU workers, each training
for 3 epochs, and then save the best checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import os

# Torch is essential for training. We import with an explicit error if missing.
try:
    import torch  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required to run training. Install via Conda or ensure 'pytorch' in env.yml."
    ) from exc

try:
    import ray
    from ray import tune
    from ray.train import Checkpoint
    from ray.train.torch import TorchTrainer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Ray with Train & Tune extras is required. Install ray[air].") from exc

from torch.utils.data import DataLoader

from python.graph.builder import build_graph
from python.model.hgt import MiniHGT
from python.data.nightly_fetch import _fake_lines, _fake_results

# -----------------------------------------------------------------------------
# Data loader helpers (placeholder – use in-memory graph for now)
# -----------------------------------------------------------------------------

def _build_dataset() -> Dict[str, torch.Tensor]:  # noqa: D401
    try:
        import ray.data  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "ray.data is required to build datasets. Ensure Ray is installed with the 'data' extra (ray[data])."
        ) from exc

    from datetime import date

    ray.init(address="auto", ignore_reinit_error=True)
    ds_lines = ray.data.from_items(_fake_lines(200, date.today()))
    ds_results = ray.data.from_items(_fake_results(100, date.today()))
    data = build_graph(ds_lines, ds_results)

    return {
        "x_dict": {k: torch.tensor(v, dtype=torch.float32) for k, v in data.x_dict.items()},
        "edge_index_dict": {k: torch.tensor(v, dtype=torch.long) for k, v in data.edge_index_dict.items()},
        "y": torch.tensor(data["game"].y, dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# Training function (executed on Ray worker)
# -----------------------------------------------------------------------------

def _train_worker(config):  # noqa: ANN001  D401
    dataset = _build_dataset()

    node_types = list(dataset["x_dict"].keys())
    edge_types = list(dataset["edge_index_dict"].keys())
    model = MiniHGT(metadata=(node_types, edge_types))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset["x_dict"], dataset["edge_index_dict"])
        loss = criterion(logits, dataset["y"])
        loss.backward()
        optimizer.step()

        tune.report(loss=loss.item())

    # Return best model.
    checkpoint = Checkpoint.from_dict({"model_state_dict": model.state_dict()})
    return checkpoint

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train MiniHGT")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick single epoch training")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of hyperparameter samples")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    config = {"lr": tune.loguniform(1e-4, 1e-2), "epochs": args.epochs}

    if args.smoke_test:
        config["lr"] = 1e-3
        args.num_samples = 1

    # If user sets --gpu flag (detected by CUDA env), validate CUDA availability.
    if os.environ.get("FORCE_GPU", "0") == "1" and not torch.cuda.is_available():
        raise EnvironmentError(
            "FORCE_GPU=1 but CUDA is not available. Install CUDA-enabled PyTorch or unset FORCE_GPU."
        )

    tuner = tune.Tuner(
        _train_worker,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=args.num_samples, metric="loss", mode="min"),
    )

    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    best_checkpoint = best_result.checkpoint

    try:
        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)
        file_path = out_dir / "best.pt"
        torch.save(best_checkpoint.to_dict()["model_state_dict"], file_path)

        # ------------------------------------------------------------------
        # Publish model URI for downstream agents (Serve + CLI).
        # ------------------------------------------------------------------
        from urllib.parse import urlparse

        prefix = os.getenv("MODEL_URI_PREFIX", "file://")  # Agent 2 may set to runpod://…
        if not file_path.exists():
            raise FileNotFoundError(f"Expected checkpoint at {file_path} not found.")

        model_uri = prefix.rstrip("/") + "/" + str(file_path.resolve())

        # Basic validation – must have scheme://
        parsed = urlparse(model_uri)
        if not parsed.scheme:
            raise ValueError(
                f"MODEL_URI_PREFIX produced an invalid URI: '{model_uri}'. Prefix must include scheme like 'file://' or 'runpod://'."
            )

        # Persist for other jobs & print for logs.
        uri_file = out_dir / "latest_uri.txt"
        with open(uri_file, "w", encoding="utf-8") as fh:
            fh.write(model_uri)

        print(f"🏆 Saved best model → {file_path.resolve()}")
        print(f"🔗 Published model URI  → {model_uri}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save/publish model checkpoint: {exc}") from exc


if __name__ == "__main__":
    main()