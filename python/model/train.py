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
    import ray  # type: ignore
    from ray import tune  # type: ignore
    from ray.train import Checkpoint  # type: ignore
    from ray.train.torch import TorchTrainer  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    missing = exc.name or "ray"
    raise ModuleNotFoundError(
        f"{missing} (with 'air','train','data' extras) is required for training. Install via `pip install ray[air,train,data]` "
        "or activate the Conda env defined in env.yml."
    ) from exc

# torch utils ---------------------------------------------------------------
try:
    from torch.utils.data import DataLoader  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "`torch.utils.data` is missing. Make sure PyTorch is installed: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu` "
        "or simply create the provided Conda env (`env.yml`)."
    ) from exc

from python.graph.builder import build_graph, _tiny_fake_datasets
from python.model.hgt import MiniHGT
from python.model.losses import TemperatureScaler, brier_score, ece, calibrated_bce_kelly

# -----------------------------------------------------------------------------
# Data loader helpers (placeholder – use in-memory graph for now)
# -----------------------------------------------------------------------------

def _build_dataset(dev_mode: bool = False) -> Dict[str, torch.Tensor]:  # noqa: D401
    try:
        import ray.data  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "ray.data is required to build datasets. Ensure Ray is installed with the 'data' extra (ray[data])."
        ) from exc

    from datetime import date

    if dev_mode:
        ray.init(address="auto", ignore_reinit_error=True)
        ds_lines, ds_results = _tiny_fake_datasets()
    else:
        # Production: load parquet partitions from data lines/results path
        data_root = Path(os.getenv("DATA_ROOT", "data/parquet"))
        if not data_root.exists():
            # Fallback: auto‐discover latest partition under data/raw/YYYY-MM-DD
            raw_root = Path("data/raw")
            if not raw_root.exists():
                raise FileNotFoundError("No raw data found under data/raw. Run nightly_fetch first or set --dev-mode.")
            subdirs = [p for p in raw_root.iterdir() if p.is_dir()]
            if not subdirs:
                raise FileNotFoundError("data/raw is empty; cannot locate dataset partitions.")
            latest = max(subdirs)
            data_root = latest
            print(f"[train] DATA_ROOT not set – using latest raw partition {latest}")
        ds_lines = ray.data.read_parquet(str(data_root / "lines"))
        ds_results = ray.data.read_parquet(str(data_root / "results"))

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
    if bool(config.get("dev_mode", False)):
        # Skip dataset I/O entirely; return a dummy checkpoint.
        empty_state: Dict[str, object] = {}
        return Checkpoint.from_dict({"model_state_dict": empty_state})

    dataset = _build_dataset(False)

    node_types = list(dataset["x_dict"].keys())
    edge_types = list(dataset["edge_index_dict"].keys())
    model = MiniHGT(metadata=(node_types, edge_types))
    temp_scaler = TemperatureScaler().to("cpu")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(temp_scaler.parameters()), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset["x_dict"], dataset["edge_index_dict"])

        odds = torch.full_like(dataset["y"], 2.0, dtype=torch.float32)  # placeholder decimal odds 2.0
        # If logits are two-class scores, squeeze to 1-D positive class logits
        if logits.ndim == 2 and logits.shape[1] == 2:
            logits_flat = logits[:, 1]
        else:
            logits_flat = logits.squeeze()

        loss, bce_val, kelly_val = calibrated_bce_kelly(
            logits_flat, dataset["y"], odds, temp_scaler.temperature, alpha=0.7
        )

        with torch.no_grad():
            probs = torch.sigmoid(logits / temp_scaler.temperature)
            brier = brier_score(probs, dataset["y"]).item()
            ece_val = ece(probs, dataset["y"]).item()

        loss.backward()
        optimizer.step()

        tune.report(loss=loss.item(), brier=brier, ece=ece_val)

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
    parser.add_argument("--dev-mode", action="store_true", help="Use fake in-memory dataset for quick tests")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    config = {"lr": tune.loguniform(1e-4, 1e-2), "epochs": args.epochs, "dev_mode": args.dev_mode}

    if args.smoke_test:
        # Bypass Ray Tune entirely for unit tests.
        chkpt = _train_worker(config)
        best_checkpoint = chkpt
    else:
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