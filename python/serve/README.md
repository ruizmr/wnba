# Edge Model Serve Endpoint

The service is deployed via Ray Serve and exposes a RESTful interface powered by FastAPI.

## Base URL

```
GET /healthz
```
Returns `200` with JSON `{ "status": "ok", "model_uri": "..." }` when the service is healthy.

```
POST /predict
Content-Type: application/json
```
Request body schema:

```jsonc
{
  "game_id": 12345,             // integer ID of the game (from data lines)
  "features": {                 // arbitrary feature map; schema TBD by Agent 1
    "home_team_rank": 17,
    "away_team_rank": 3
  }
}
```

Response body schema:

```jsonc
{
  "game_id": 12345,
  "win_prob": 0.73              // model-predicted probability that `game_id` home team wins.
}
```

## Observability

Prometheus metrics are exposed at:

```
GET /metrics
```

The scrape includes:

* `predict_requests_total{method="POST", http_status="200"}` – counter
* `predict_latency_seconds` – histogram buckets 10 ms → 5 s

Example scrape:

```text
# HELP predict_requests_total Number of /predict requests processed
# TYPE predict_requests_total counter
predict_requests_total{method="POST",http_status="200"} 42
# HELP predict_latency_seconds Latency of /predict endpoint in seconds
# TYPE predict_latency_seconds histogram
predict_latency_seconds_bucket{le="0.1"} 40
...
```

## Usage Examples

```bash
# Local dev (CPU)
ray start --head &
export MODEL_URI=models/best.pt
python -m serve.app

# Query
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"game_id": 1, "features": {}}'
```

## Deployment on RunPod

The cluster is defined in `.ray/cluster.yaml`.  Spin it up:

```bash
ray up -y .ray/cluster.yaml        # create head & GPU workers
ray rsync-up . .                   # sync source code
ray submit .ray/cluster.yaml python -m serve.app --start
```

The external URL will be printed by the autoscaler once the Serve HTTP proxy is ready.  Add that URL to the `PREDICT_URL` environment variable for Agent 3 consumers.

### Tracing (OpenTelemetry)

On startup the service auto-instruments FastAPI via OTLP HTTP exporter.
Set environment variable before launching:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector:4318/v1/traces
```

Spans are tagged with `service.name = edge-serve`.  Example dashboard: Grafana Tempo + Grafana Loki.

### AuthN

`/predict` is protected by JSON Web Tokens (JWT).  Clients must include:

```
Authorization: Bearer <token>
```

Tokens are signed with HS256.  Configure the shared secret via env:

```bash
export JWT_SECRET="replace-with-32-byte-secret"
```

Generate token (Python):

```python
import jwt, time, os
payload = {"sub": "agent3", "exp": int(time.time()) + 3600}
print(jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256"))
```

`/healthz` and `/metrics` remain public for liveness & scraping.

### Rate limiting

All `/predict` calls are limited to `60/minute` per IP by default.

Override via env var `RATE_LIMIT_TIER` in Serve deployment (planned) or edit
`serve/app.py` decorator.  In production, set `REDIS_URL` so limits are shared
across replicas.