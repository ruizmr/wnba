# pyright: reportMissingImports=false
"""Rate-limiting middleware using *slowapi* (Redis optional).

`slowapi` wraps the `limits` package.  We default to an in-memory limiter which
is process-safe for a single-replica deployment.  If the environment variable
`REDIS_URL` is set, we switch to a Redis storage backend (recommended for multi-
replica clusters).

Example
-------
from serve.rate_limit import init_rate_limit

app = FastAPI()
limiter = init_rate_limit(app)

@app.get("/ping")
@limiter.limit("5/minute")
async def ping():
    return "pong"
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def _create_storage() -> Any:  # pragma: no cover
    """Return a storage backend for slowapi.limiter depending on env."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from limits.storage import RedisStorage  # lazy import

        logger.info("Rate limit storage: Redis → %s", redis_url)
        return RedisStorage(redis_url)

    from limits.storage import MemoryStorage

    logger.warning("REDIS_URL not set; using in-memory rate limiting.")
    return MemoryStorage()


def init_rate_limit(app: FastAPI) -> Limiter:  # noqa: D401
    """Attach slowapi limiter to *app* and return the instance."""
    limiter = Limiter(key_func=get_remote_address, storage=_create_storage())
    app.state.limiter = limiter  # type: ignore[attr-defined]
    app.add_exception_handler(429, _rate_limit_exceeded_handler)
    return limiter