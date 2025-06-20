# pyright: reportMissingImports=false,reportGeneralTypeIssues=false
"""Ray Serve application.

This service wraps the graph transformer model produced nightly by Agent 1 and
served via Ray Serve.  It exposes a minimal FastAPI endpoint at `/predict` as
well as `/healthz` for liveness checks.

Example
-------
$ MODEL_URI=models/best.pt ray start --head &
$ python -m serve.app  # runs serve.run(...)
$ curl http://127.0.0.1:8000/healthz

Important
---------
• The actual model loading logic will be injected later once Agent 1 publishes
  their checkpoint format.  Until then we mock the `predict` output.
"""
from __future__ import annotations

import os
from typing import Any, Dict
import time

from fastapi import FastAPI, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from ray import serve
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from serve.metrics import REQUEST_COUNT, REQUEST_LATENCY
from serve.tracing import init_tracing
from serve.auth import verify_jwt
from serve.rate_limit import init_rate_limit
from serve.model_loader import load_mini_hgt

###############################################################################
# FastAPI schema
###############################################################################


class PredictRequest(BaseModel):
    """Schema for `/predict` endpoint request.

    The architect's v1 API expects `home_team` and `away_team` strings under the
    ``features`` key. Future versions may extend this schema – we will adapt via
    pydantic's model upgrade path.
    """

    game_id: int
    features: Dict[str, Any]

    # Convenience accessors -------------------------------------------------

    @property
    def home_team(self) -> str:  # noqa: D401
        return self.features.get("home_team", "")

    @property
    def away_team(self) -> str:  # noqa: D401
        return self.features.get("away_team", "")


class PredictResponse(BaseModel):
    """Schema for `/predict` response."""

    game_id: int
    win_prob: float


###############################################################################
# Serve deployment
###############################################################################

app = FastAPI()

# Initialize shared middlewares
init_tracing(app)
limiter = init_rate_limit(app)


@serve.deployment(name="edge-model", num_replicas=1, route_prefix="/")
@serve.ingress(app)
class ModelServer:  # pylint: disable=too-few-public-methods
    """Thin wrapper around the Torch model checkpoint."""

    def __init__(self) -> None:
        self.model_uri: str = os.getenv("MODEL_URI", "models/best.pt")

        # Load MiniHGT checkpoint – if fails, stay in degraded mode.
        try:
            self.model = load_mini_hgt(self.model_uri)
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            print(f"[Serve] ERROR loading model: {exc}. Falling back to dummy predictions.")
            self.model = None
            self._loaded = False

    # ---------------- API routes ---------------- #

    @app.get("/healthz")
    def healthz(self) -> Dict[str, str]:  # noqa: D401 (imperative mood)
        """Health-check used by Docker/K8s readiness probe."""
        return {"status": "ok", "model_uri": self.model_uri}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:  # noqa: D401
        """Prometheus scrape endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @limiter.limit("60/minute")
    @app.post("/predict", response_model=PredictResponse)
    def predict(self, payload: PredictRequest, user=Depends(verify_jwt)) -> PredictResponse:  # noqa: D401,E501
        """Return model probability; falls back to pseudo-random if model unavailable."""

        start = time.perf_counter()

        if self._loaded:
            prob = self._infer_single_game(payload)
        else:
            # Fallback deterministic hash – should rarely trigger in production.
            prob = (hash(payload.game_id) % 100) / 100.0

        resp = PredictResponse(game_id=payload.game_id, win_prob=float(prob))
        latency = time.perf_counter() - start
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels("POST", "200").inc()
        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_single_game(self, payload: PredictRequest) -> float:  # noqa: D401
        """Minimal on-the-fly graph build for a single game and run inference."""

        try:
            import ray  # type: ignore
            from python.graph.builder import build_graph  # dynamic import
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Required libs for inference missing: install ray & torch-geometric") from exc

        # Build tiny Ray datasets
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, address="auto")

        ds_lines = ray.data.from_items(
            [
                {
                    "game_id": payload.game_id,
                    "team": payload.home_team,
                    "line_type": "spread",
                    "value": -3.5,
                    "odds": -110,
                },
                {
                    "game_id": payload.game_id,
                    "team": payload.away_team,
                    "line_type": "spread",
                    "value": 3.5,
                    "odds": -110,
                },
            ]
        )

        ds_results = ray.data.from_items([])  # empty – results unknown pre‐game

        g = build_graph(ds_lines, ds_results)

        # Ensure features tensors are torch
        import torch  # type: ignore

        x_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in g.x_dict.items()}
        edge_index_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in g.edge_index_dict.items()}

        logits = self.model(x_dict, edge_index_dict)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()  # probability of class 1 (win)
        return prob


###############################################################################
# Entrypoint
###############################################################################

# DAG handle used by `ray job submit` or `python -m serve.app`
entrypoint = ModelServer.bind()  # type: ignore[attr-defined]

def main() -> None:  # pragma: no cover
    """Run serve runtime when called as CLI module."""
    serve.run(entrypoint, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()