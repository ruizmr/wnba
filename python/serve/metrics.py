# pyright: reportMissingImports=false
"""Prometheus metrics for Edge Serve.

This module centralizes Prometheus counters / histograms so they can be imported
across files without causing duplicate registrations.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Number of /predict requests processed",
    ["method", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency of /predict endpoint in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)