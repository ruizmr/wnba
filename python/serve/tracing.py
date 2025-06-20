# pyright: reportMissingImports=false
"""OpenTelemetry tracing setup for Edge Serve.

This module initializes OTLP exporter and FastAPI instrumentation.  Importing
this module has side effects which register middleware, so it should be called
exactly once at startup (e.g. in `serve.app`).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"


def init_tracing(app) -> None:  # type: ignore[valid-type]
    """Initialize OpenTelemetry tracing and attach middleware to *app*.

    Parameters
    ----------
    app : fastapi.FastAPI
        The FastAPI application object.
    """
    if trace.is_tracer_provider_set():
        logger.info("TracerProvider already configured; skipping init.")
        return

    endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)

    resource = Resource(attributes={SERVICE_NAME: "edge-serve"})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    span_processor = BatchSpanProcessor(span_exporter)
    provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)

    # Instrument FastAPI (ASGI) with OpenTelemetry middleware.
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    # For non-FastAPI middlewares, instrumentation.asgi covers generic cases.
    if not any(isinstance(m, OpenTelemetryMiddleware) for m in app.user_middleware):
        app.add_middleware(OpenTelemetryMiddleware)

    logger.info("OpenTelemetry tracing initialized → %s", endpoint)