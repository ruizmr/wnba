# Grafana Dashboards for Edge Serve

This folder contains the JSON definitions that can be imported into Grafana.

* `edge_serve_dashboard.json` — Overview dashboard with request rate and p95 latency.

## Import Steps

1. Log in to Grafana (`http://localhost:3000` or your managed instance).
2. Side-bar → **Dashboards** → **Import**.
3. Upload the JSON file or paste its contents.
4. Select your Prometheus data-source when prompted.

Alternatively, use the Grafana API:

```bash
curl -X POST -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GRAFANA_API_KEY" \
     -d @edge_serve_dashboard.json \
     http://grafana:3000/api/dashboards/db
```

Once imported, you'll see real-time charts populate as soon as Prometheus begins scraping the `/metrics` endpoint on Serve pods.