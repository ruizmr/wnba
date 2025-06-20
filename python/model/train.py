"""Training entrypoint for the `MiniHGT` model.

This script is designed to be launched via Ray Jobs or `ray submit` as required
by Agent 2. It supports local CPU execution for quick smoke tests and Tune-based
hyperparameter search when running on a multi-GPU cluster.

Example (local CPU)
-------------------
ray job submit --runtime-env="{}" python python/model/train.py --smoke-test

Example (RunPod GPU cluster)
---------------------------
ray submit .ray/cluster.yaml python/model/train.py --num-samples 8 --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

try:
    import ray
    from ray import tune
    from ray.train import Checkpoint
    from ray.train.torch import TorchTrainer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Ray with Train & Tune extras is required. Install ray[air].") from exc

import torch
from torch.utils.data import DataLoader

from python.graph.builder import build_graph
from python.model.hgt import MiniHGT
from python.data.nightly_fetch import _fake_lines, _fake_results

# -----------------------------------------------------------------------------
# Data loader helpers (placeholder – use in-memory graph for now)
# -----------------------------------------------------------------------------

def _build_dataset() -> Dict[str, torch.Tensor]:  # noqa: D401
    import ray.data
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

    tuner = tune.Tuner(
        _train_worker,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=args.num_samples, metric="loss", mode="min"),
    )

    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    best_checkpoint = best_result.checkpoint

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / "best.pt"
    torch.save(best_checkpoint.to_dict()["model_state_dict"], file_path)

    print(f"🏆 Saved best model to {file_path.resolve()}")


if __name__ == "__main__":
    main()