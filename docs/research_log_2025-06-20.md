# Research Log – 2025-06-20

This note captures the June-20 literature injection (BB-tags) for Phase-2.

| Paper | Why it matters | BB-Tag |
|-------|---------------|--------|
| Kuleshov et al., 2018 (ICML) | Provides temperature & Dirichlet calibration baselines that lower ECE – prerequisite for CalibratedKellyLoss. | BB-L1 |
| Ma et al., 2023 (ICML) | Shows how *synth* edge buckets inject heterophilous inductive bias into Transformers. We will port the relative-angle buckets into `graph/builder.py`. | BB-D1 |
| Rahimi & Beck 2023 | Dirichlet calibration head consistently beats Temp-Scaling on OOD splits – may replace / augment Edge-29. | BB-L1 |
| Vélez 2023 | Convex Kelly-matrix solver with closed-form dual. Implementation path: JAX → export weights. | BB-M1 |
| Zhang 2024 | CVaR-aware utility shaping outperforms plain Kelly in heavy-tail regimes – supports planned `RiskAwareReward` (Edge-78). | BB-L3 |

Next actions
-------------
1. Draft PR to add DirichletCalibrator module.
2. Prototype synth-edge generation on small subset – measure model latency impact.
3. JAX spike for convex Kelly; benchmark vs existing PyTorch solver.
4. Design experiment matrix for CVaR-Kelly loss.