from __future__ import annotations

"""Central configuration for the Edge project.

All runtime‐tuneable options are exposed as environment variables so that
Docker/Helm and local workflows remain in sync.  Access via the singleton
`settings` instance.
"""
from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Project settings loaded from environment variables."""

    # ------------------------------------------------------------------
    # Serve runtime
    # ------------------------------------------------------------------
    model_uri: str = Field("models/best.pt", env="MODEL_URI")
    enable_fallback: bool = Field(False, env="ENABLE_FALLBACK")

    # Authentication
    jwt_secret: str = Field("super-secret", env="JWT_SECRET")

    # Rate limiting (SlowAPI syntax, e.g. "60/minute")
    rate_limit: str = Field("60/minute", env="RATE_LIMIT")

    class Config:  # pylint: disable=too-few-public-methods
        case_sensitive = False
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # noqa: D401
    """Return singleton settings object."""

    return Settings() 