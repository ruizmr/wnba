══════════════════════════════════════
PHASE-2 ROADMAP (Weeks 2-4)
══════════════════════════════════════
The MVP from *agents.md* is live.  Phase-2 focuses on scaling, profitability, and research acceleration.  Four autonomous agents own the vertical slices below.  All tasks are Jira-linked (e.g., EDGE-42) and follow "feat/fix/docs/chore" Conventional Commits.


══════════════════════════════════════
AGENT 1  "Architect / Graph & Model v2"
══════════════════════════════════════
North-Star   Turn the MiniHGT baseline into a calibrated, Kelly-aware powerhouse and prep for multi-league transfer.

Milestones
1. **Edge-21** `data/multileague_pbp_datamodule.py` – load NBA, NCAA-W, WNBA PBP into unified schema → Parquet partitions (use Arrow Dataset API).
2. **Edge-25** `graph/builder.py` – add **Vegas line** & **line-move edges** (*Action BB-D1* from research_summary.md).
3. **Edge-29** `model/hgt.py` – inject **temperature scaling head** + expose `forward(..., return_logits=False)`.
4. **Edge-30** `model/losses.py` – implement **CalibratedKellyLoss** and **ECE/Brier metrics** (*BB-L1/L2*).
5. **Edge-34** `model/train.py` – Tune sweep: λ_kelly∈{0,0.05,0.1}, lr, hidden_dim; push best.pt nightly.

Definition of Done
✓ Tune job converges on A5000 GPUs in < 3 hrs.  nECE < 0.03 on validation.
✓ CI artifact uploads `models/best.pt` + `metrics.json`.


══════════════════════════════════════
AGENT 2  "DevOps / Runtime & Observability"
══════════════════════════════════════
North-Star   Productionise the pipeline with zero-touch deploys and bullet-proof monitoring.

Milestones
1. **Edge-41** GitHub Actions matrix (CPU + CUDA) → builds Docker, runs pytest, uploads coverage to Codecov.
2. **Edge-42** `Dockerfile` harden: non-root user, multi-stage build, layer cache.
3. **Edge-45** `infra/` – Helm chart `edge-serve`;  readiness probe `/healthz`, Prometheus scrape `/metrics`.
4. **Edge-48** Grafana dashboard (extend `grafana/edge_serve_dashboard.json`) with p99 latency, error-rate, ece gauge.
5. **Edge-50** Cron jobs as **Ray Jobs API** → move bash wrappers inside `.github/workflows/schedule.yml`.

Definition of Done
✓ Helm install on RunPod returns green. Grafana dashboard shows live metrics for test traffic.


══════════════════════════════════════
AGENT 3  "App Dev & Risk / Lenses + Reporting v2"
══════════════════════════════════════
North-Star   Maximise bankroll growth & transparency for end-users.

Milestones
1. **Edge-55** `lenses.py` – add **Kelly-fraction lens** (uses model win_prob & decimal odds).
2. **Edge-58** `aggregate.py` – implement learned weights via Ridge regression, cache coefficients.
3. **Edge-60** `cli.py` – new sub-cmd `predict ledger` → pretty table of upcoming slate with stake sizes.
4. **Edge-64** `report/daily.py` – overlay bankroll chart, Sharpe, Sortino (*research_summary §2*).
5. **Edge-66** End-to-end smoke test: `cli predict slate --date 2025-06-01` → writes ledger row in dry-run.

Definition of Done
✓ Snapshot diff of HTML report stable; Sortino ≥ 2.0 on back-test sample.


══════════════════════════════════════
AGENT 4  "Big-Brain / Research & Data Enrichment"
══════════════════════════════════════
North-Star   Continuously inject state-of-the-art ideas & datasets to keep the edge ahead of the market.

Milestones
1. **Edge-70** Update `docs/research_summary.md` weekly; tag actionable bullets (BB-L*, BB-D*, BB-M*).
2. **Edge-72** `data/fetch_*_stats.py` – finish NBA & NCAA-W scrapers; include bookmaker margin (*BB-M1*).
3. **Edge-75** Prototype **convex Kelly-matrix solver** (see Vélez 2023) and benchmark inside `notebooks/kelly_matrix.ipynb`.
4. **Edge-78** Design experiment to test RL risk reward (stub `losses.py::RiskAwareReward`) (*BB-L3*).
5. **Edge-80** Run ablation: calibration vs ROI; publish findings in `reports/research_memo_YYMMDD.md`.

Definition of Done
✓ Every research action produces a GitHub Issue with summary + PR link.
✓ Reading list in BibTeX grows > 30 curated papers.


══════════════════════════════════════
SHARED NORMS (Phase-2 additions)
══════════════════════════════════════
• Every PR must tick a *research linkage* checkbox if it implements a BB-tag.
• 95 % coverage target for pure math utils; 85 % overall.
• Merge gate: `pytest -q` + `pre-commit` + `ray job submit --dry-run`.
• Slack #edge-alerts channel: Serve error-rate spikes > 1 % triggers message & on-call rotation (Agent 2 primarily). 