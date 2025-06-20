# pyright: reportMissingImports=false
"""Utility to load MiniHGT checkpoint via URI.

Supports `file://`, `s3://`, `http://`, and `https://` using *fsspec* if
installed. Falls back to local filesystem for `file://` and plain paths.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple, Any

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _resolve_uri(uri: str) -> Path:
    """Download *uri* to a local temp file if remote, else return local path."""

    from urllib.parse import urlparse
    import tempfile

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path)

    # Remote – try fsspec.
    try:
        import fsspec  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Remote URI scheme '{parsed.scheme}' requires fsspec: pip install fsspec[s3]"  # noqa: E501
        ) from exc

    fs, _ = fsspec.core.url_to_fs(uri)
    local_tmp = Path(tempfile.mkstemp(suffix=".pt")[1])
    with fs.open(uri, "rb") as src, local_tmp.open("wb") as dst:
        dst.write(src.read())
    logger.info("Downloaded remote checkpoint %s → %s", uri, local_tmp)
    return local_tmp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_mini_hgt(uri: str, device: str = "cpu"):
    """Load MiniHGT checkpoint at *uri* on *device* ('cpu' or 'cuda')."""

    if torch is None:
        raise RuntimeError("PyTorch not available; cannot load model.")

    ckpt_path = _resolve_uri(uri)

    # Dynamic import to avoid hard dep when architect code not yet merged.
    try:
        from importlib import import_module

        MiniHGT = getattr(import_module("python.model.hgt"), "MiniHGT")  # type: ignore[attr-defined]
    except (ModuleNotFoundError, AttributeError):  # pragma: no cover
        raise RuntimeError(
            "MiniHGT architecture not found. Ensure architect code is merged under python/model/hgt.py"
        )

    # Naïve metadata assumption; architect may revise – we will sync when they do.
    node_types = ["team", "game"]
    edge_types = [("team", "participates", "game")]

    model = MiniHGT(metadata=(node_types, edge_types))
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded MiniHGT checkpoint from %s (device=%s)", ckpt_path, device)
    return model