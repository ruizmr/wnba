# Edge Engine – MVP

This repository contains three cooperating "agents" that together build a daily
basketball edge-finding pipeline.

## Quick start (local CPU)
```bash
# Clone & enter repo
conda env create -f env.yml          # installs PyTorch + PyG + Ray + tooling
conda activate edge-engine

# Run unit tests
pytest -q

# Generate fake datasets & Parquet partitions
python -m python.data.nightly_fetch --date 2024-01-01 --rows 100

# Build graph & cache it
python -m python.graph.builder \
    --lines data/raw/2024-01-01/lines \
    --results data/raw/2024-01-01/results \
    --out data/graph_cache/2024-01-01.pt

# Train a tiny HGT model (single-trial smoke test)
ray job submit --runtime-env="{}" \
    python python/model/train.py --smoke-test --epochs 1
```

## Running on RunPod GPU cluster
1. Ensure your Ray cluster is defined at `.ray/cluster.yaml` (Agent 2's job).  
2. SSH or `ray submit` the training script:
```bash
ray submit .ray/cluster.yaml python/model/train.py --num-samples 8 --epochs 3
```
Environment variables:
* `MODEL_URI_PREFIX` – prefix inserted before the absolute path when
  `train.py` publishes its `models/latest_uri.txt`.  
  Example for RunPod object store: `export MODEL_URI_PREFIX="runpod://edge-bucket"`.

## Graph cache contract
* Latest graph snapshot is stored under `data/graph_cache/<YYYY-MM-DD>.pt` (or
  `data/graph_cache/latest.pt` depending on the job).  
* Agents 2 & 3 can load it with `torch.load()`.

## Model URI contract
* After training, `python/model/train.py` writes:  
  * `models/best.pt` – binary model weights.  
  * `models/latest_uri.txt` – single-line URI (e.g. `file:///…/best.pt` or
    `runpod://…/best.pt`)
* The Serve deployment (Agent 2) should read this text file to discover the
  newest checkpoint.

---
For further details see `agents.md`.

> **Note**
> `env.yml` already includes **PyArrow**, which Ray Data needs for Parquet I/O.
> GPU users can leave the bundled CUDA-11.8 packages in place, while CPU-only
> developers may comment-out the three CUDA-labelled lines if Conda cannot find
> a compatible toolkit on their machine.

## Development setup
Every contributor should follow **one** of the two patterns below so that local
`import` statements resolve immediately in editors, notebooks and CLI tools.

**Step-by-step**

1. Clone the repo and move into its directory.  
2. Create the Conda environment (once):

   ```bash
   conda env create -f env.yml
   ```

3. Activate the env on every new shell:

   ```bash
   conda activate edge-engine
   ```

4. Choose one of:

   • **Editable install** (recommended for day-to-day coding):

   ```bash
   pip install -e .        # changes take effect instantly
   ```

   • **PYTHONPATH hack** (quick & dirty, no wheel install):

   ```bash
   export PYTHONPATH=$(pwd)
   ```

If your IDE still reports "import not found", double-check that it is using
the `edge-engine` interpreter.

## Common "import not found" fixes
| Missing package | One-liner fix |
|-----------------|---------------|
| torch           | `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu` |
| torch_geometric | `pip install torch-geometric` *(plus scatter/sparse wheels)* |
| ray             | `pip install "ray[default,air,data,train,serve]"` |
| numpy           | `pip install numpy` |
| pydantic        | `pip install pydantic` |

If you **did** run `conda env create -f env.yml` these should already be present.

```bash