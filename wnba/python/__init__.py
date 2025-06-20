from importlib import import_module

# Lazy import to avoid boto3 cost unless required

def open_artifact(uri: str, mode: str = "rb"):
    """Convenience wrapper around utils.artifact.open."""
    artifact = import_module("wnba.python.utils.artifact")
    return artifact.open(uri, mode)

__all__ = [
    "open_artifact",
]