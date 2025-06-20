from __future__ import annotations

import io
import os
import pathlib
from urllib.parse import urlparse
from typing import Union, BinaryIO, TextIO

try:
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # lazy optional dep

__all__ = ["open"]


class UnsupportedSchemeError(ValueError):
    """Raised when an artifact scheme is recognised but not supported yet."""


class MissingDependencyError(RuntimeError):
    """Raised when a runtime dependency is required but not installed."""


Reader = Union[BinaryIO, TextIO]


def _open_local(parsed, mode: str) -> Reader:
    path = pathlib.Path(parsed.path)
    # Ensure parent directory exists on write
    if any(c in mode for c in ("w", "a", "+")):
        path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)  # type: ignore[arg-type]


def _open_s3(parsed, mode: str) -> Reader:  # pragma: no cover
    if boto3 is None:
        raise MissingDependencyError("boto3 is required for S3 support")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")

    if "r" in mode and "+" not in mode and "w" not in mode:
        # read-only path returns BytesIO
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return io.BytesIO(body)

    raise UnsupportedSchemeError("S3 write support not implemented yet")


def open(uri: str, mode: str = "rb") -> Reader:  # noqa: A001
    """Open an artifact URI and return a file-like object.

    Currently supports:
        • Local filesystem: artifact:///absolute/path/to/file or plain paths
        • S3 (read-only):   artifact+s3://bucket/key
    """
    if uri.startswith("artifact://"):
        parsed = urlparse(uri)
        return _open_local(parsed, mode)

    if uri.startswith("artifact+s3://"):
        parsed = urlparse(uri.replace("artifact+", ""))
        return _open_s3(parsed, mode)

    # Fallback to treating as plain filesystem path
    return _open_local(urlparse(f"file://{os.path.abspath(uri)}"), mode)