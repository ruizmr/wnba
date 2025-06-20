# RunPod + Ray Guide

This short note explains how to spin up the **edge-engine** cluster on RunPod Cloud GPUs and keep data-fetch jobs on your CPU dev box.

---
## 1. Prerequisites

1. Create a RunPod account and generate an **API key** (`RUNPOD_API_KEY`).  
   Dashboard → *API Keys* → *New Key*.
2. Install the Ray CLI ≥ 2.9 locally:
   ```bash
   pip install "ray[default]==2.9.0"
   ```
3. Install the RunPod provider plugin for Ray autoscaler:
   ```bash
   pip install runpod-ray-launcher
   ```

---
## 2. Launch the Ray Cluster on RunPod

The repo ships with `.ray/cluster.yaml` which requests one **CPU head** node and auto-scaling **RTX A5000** GPU workers (0–4).

```bash
export RUNPOD_API_KEY="<your-key>"
ray up .ray/cluster.yaml -y   # ~2–3 min until head is ready
ray status .ray/cluster.yaml  # watch until cluster is healthy
```

After that Ray prints a **dashboard URL** and `ray://<IP>:10001` address; copy the address — we'll reference it as `$RUNPOD_ADDR` below.

---
## 3. Data-fetch vs Training Separation

| Job | Where it runs | How it's launched |
|-----|---------------|-------------------|
| `python.data.nightly_fetch` | **Local CPU box** (this repo) | via GitHub Action *Nightly Pipeline → fetch_lines* OR manual `python -m python.data.nightly_fetch` |
| `python.model.train` | **RunPod GPU cluster** | `ray job submit --address $RUNPOD_ADDR -- python -m python.model.train --epochs 3`|
| `python.report.daily` | **Local box** (cheap) | `ray job submit --address http://127.0.0.1:8265 …` |

The scheduled GitHub workflow (`.github/workflows/schedule.yml`) already follows this pattern:
* Uses `secrets.RAY_ADDRESS` for *fetch* & *train* jobs — set this to `$RUNPOD_ADDR` in repository secrets.
* Training job runs **after** fetch completes, so fresh Parquet partitions are available.

If you prefer running fetch locally outside CI, simply keep the workflow disabled (comment out) and run the script in a cron on your machine.

---
## 4. Serving with Helm

1. **Build & Push** container:
   ```bash
   IMAGE=ghcr.io/<user>/edge-serve:$(git rev-parse --short HEAD)
   docker build -t $IMAGE -f Dockerfile --target runtime .
   docker push $IMAGE
   ```
2. **Edit** `infra/edge-serve/values.yaml`:
   ```yaml
   image:
     repository: ghcr.io/<user>/edge-serve
     tag: <sha>
   env:
     JWT_SECRET: "$(openssl rand -hex 16)"
   ```
3. **Install Helm** (one-liner): `curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash`
4. **Deploy**:
   ```bash
   helm upgrade --install edge-serve infra/edge-serve -f infra/edge-serve/values.yaml
   kubectl get pods -l app.kubernetes.io/name=edge-serve -w
   ```
5. **Validate**:
   ```bash
   kubectl port-forward svc/edge-serve 8000:8000 &
   curl http://127.0.0.1:8000/healthz
   ```
   The endpoint should return 200 with `model_uri` once the first training job has saved `models/best.pt`.

---
## 5. Troubleshooting Tips

• `ray attach .ray/cluster.yaml` lets you SSH into the RunPod head for logs.  
• Ray job logs available via Ray Dashboard at `$RUNPOD_ADDR`.
• Helm release status: `helm status edge-serve`.

---
**That's it!**  With the API key and `secrets.RAY_ADDRESS` configured, CI will fetch, train and roll new models automatically each night. 