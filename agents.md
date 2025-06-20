
══════════════════════════════════════
GLOBAL NORTH STAR
══════════════════════════════════════
Nightly cron ➜ Ray Data pulls lines/results ➜ Graph builder ➜ Ray Train hyper-search ➜ Ray Serve endpoint ➜ CLI applies Buffett lenses ➜ Generates & e-mails “Daily Edge Report”.

Everything lives in one Conda env (`env.yml`) and deploys on a RunPod Ray cluster.

══════════════════════════════════════
AGENT 1 “Architect / Graph & Model”
══════════════════════════════════════
High-level goal Own every artifact that touches tensors: data schemas, graph builder, HGT model, Ray Train.

Key directories  
• `python/data/`         – ingest → Parquet  
• `python/graph/`        – builder, utils  
• `python/model/`        – `hgt.py`, `train.py`

Week-1 deliverables  
1. `schema.py` Pydantic classes for LineRow, ResultRow (unit tests incl.).  
2. `nightly_fetch.py` Ray Data job that writes example Parquet partitions (local file-system first, S3 later).  
3. `builder.py` Function `build_graph(ds_lines, ds_results) → HeteroData` + sanity test (≥ N nodes, labels).  
4. `hgt.py` Minimal Heterogeneous Graph Transformer (torch-geometric).  
5. `train.py` Ray TorchTrainer + Tune sweep over 3 hyper-parameters; saves `best.pt`.  
6. Docstring “How to run on CPU vs RunPod GPU”.

Definition of Done  
✓ `pytest -q` passes for all of the above.  
✓ Training script finishes one epoch on CPU in < 10 min.

Inter-agent contracts  
• Publishes latest graph `.pt` to `data/graph_cache/`.  
• Publishes `models/best.pt` (URI string) to `runpod://…` once per night.

══════════════════════════════════════
AGENT 2 “DevOps / Automation & Runtime”
══════════════════════════════════════
High-level goal Everything that glues pieces together and runs on a schedule.

Key directories  
• `env.yml`, `Dockerfile`, `.ray/cluster.yaml`  
• `scripts/` (cron bash wrappers)  
• `python/serve/` (Ray Serve)  

Week-1 deliverables  
1. `env.yml`&nbsp;– reproducible Conda spec; `conda env create -f env.yml` works on Mac & RunPod.  
2. `.ray/cluster.yaml` configured for RunPod: head = CPU, workers = `GPU-RTXA5000-24GB` (autoscale 0–4).  
3. `Dockerfile` that installs env, copies repo, sets entrypoint.  
4. `serve/app.py` Ray Serve deployment reading `$MODEL_URI` env var.  
5. `cron_fetch.sh`, `cron_train.sh`, `cron_report.sh`  
   • Use `ray job submit` to run the Python tasks.  
6. GitHub Action (`.github/workflows/ci.yml`) that does `conda env create`, `pytest`, `ray job submit --dry-run`.

Definition of Done  
✓ `docker-compose up` brings up Ray Serve locally and `/healthz` returns 200.  
✓ `ray submit .ray/cluster.yaml python/model/train.py --epochs 1 --smoke-test` exits 0 on RunPod.

Inter-agent contracts  
• Exposes Serve endpoint URL + JSON schema in `serve/README.md`.  
• Notifies Agent 3 nightly via GitHub Issue or Slack that a new checkpoint is live.

══════════════════════════════════════
AGENT 3 “Application Dev & QA / Lenses + Reporting”
══════════════════════════════════════
High-level goal Turn predictions into stake suggestions, ledger entries, and the HTML email.

Key directories  
• `python/lenses.py`, `python/aggregate.py`  
• `python/report/` (daily.py, email.py)  
• `python/cli.py` (Typer)

Week-1 deliverables  
1. `lenses.py` five functions + unit tests (edge-case coverage).  
2. `aggregate.py` `geo_mean()` + placeholder for learned weights.  
3. `cli.py`   a) `predict game` b) `predict slate`.  
4. `ledger.csv` schema + append helper (date, game_id, stake, pnl).  
5. `report/daily.py` Jinja template → HTML; `email.py` sends via SMTP env vars.  
6. Rich-table pretty printing in CLI.

Definition of Done  
✓ `python -m cli predict game --test` returns table using mocked Serve response.  
✓ `pytest` snapshot test confirms HTML matches baseline.  
✓ Example ledger row appended when CLI runs in `--dry-run` mode.

Inter-agent contracts  
• Requires Serve endpoint from Agent 2 (env `PREDICT_URL`).  
• Consumes Parquet lines dataset written by Agent 1.  
• Emits `ledger.csv` and `daily_report_YYYY-MM-DD.html` in `reports/`.

══════════════════════════════════════
SHARED WORKFLOW & NORMS
══════════════════════════════════════
Version control  
• `main` protected; all work on `feat/<agent>/<topic>`; PR → review by another agent.  
• Conventional Commits (`feat:`, `fix:`, `docs:`).

Coding style  
• Black + isort for Python; pre-commit hook in repo.  
• 90 % test coverage target for pure-function modules (`lenses`, `schema`, etc.).  
• Every module must have type hints and a docstring “Example”.

Communication  
• GitHub Projects board with columns Backlog → In-Progress → Review → Done.  
• Daily stand-up thread (async) tagging blockers.  
• Agents ping each other via Issue when an interface changes (e.g., Serve JSON).

Environment  
• All agents use the same Conda env.  
• Ray cluster config owned by Agent 2; others treat Ray like a SaaS.

MVP timeline (calendar-days)  
D0 Repo scaffold pushed (Agents 2 & 3 review).  
D1 Schema + env.yml done (Agent 1 & 2).  
D3 Graph builder + lenses pass tests.  
D5 Serve endpoint and CLI `predict game` wired (end-to-end smoke demo).  
D7 First automated “Daily Edge Report” emailed (MVP complete).


The agents now have unambiguous ownership, inputs, outputs, and a seven-day critical path to a working, automated edge engine.