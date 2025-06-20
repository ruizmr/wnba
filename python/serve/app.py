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

###############################################################################
# FastAPI schema
###############################################################################


class PredictRequest(BaseModel):
    """Schema for `/predict` endpoint request."""

    game_id: int
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    """Schema for `/predict` response."""

    game_id: int
    win_prob: float


###############################################################################
# Serve deployment
###############################################################################

app = FastAPI()

# Initialize OpenTelemetry tracing (no-op if already configured)
init_tracing(app)


@serve.deployment(name="edge-model", num_replicas=1, route_prefix="/")
@serve.ingress(app)
class ModelServer:  # pylint: disable=too-few-public-methods
    """Thin wrapper around the Torch model checkpoint."""

    def __init__(self) -> None:
        self.model_uri: str = os.getenv("MODEL_URI", "models/best.pt")
        # TODO: Replace with torch.load + model.eval() when Agent 1 lands.
        print(f"[Serve] Starting with MODEL_URI={self.model_uri}")

    # ---------------- API routes ---------------- #

    @app.get("/healthz")
    def healthz(self) -> Dict[str, str]:  # noqa: D401 (imperative mood)
        """Health-check used by Docker/K8s readiness probe."""
        return {"status": "ok", "model_uri": self.model_uri}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:  # noqa: D401
        """Prometheus scrape endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(self, payload: PredictRequest, user=Depends(verify_jwt)) -> PredictResponse:  # noqa: D401,E501
        """Return dummy probability; requires valid JWT."""
        start = time.perf_counter()
        # Dummy logic: pseudo-random deterministic.
        prob = (hash(payload.game_id) % 100) / 100.0
        resp = PredictResponse(game_id=payload.game_id, win_prob=prob)
        latency = time.perf_counter() - start
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels("POST", "200").inc()
        return resp


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