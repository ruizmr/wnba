import os
from pathlib import Path

import pytest

try:
    import ray
except ModuleNotFoundError:
    ray = None  # type: ignore

# Skip entire module if Ray is missing
pytestmark = pytest.mark.skipif(ray is None, reason="ray not installed")

WEHOOP_BASE = Path(os.getenv("WEHOOP_DATA_PATH", "/tmp/wehoop/wnba"))

if ray is not None:
    from python.data.wehoop_ingest import load_wehoop
else:
    load_wehoop = None  # type: ignore


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    # Only start Ray if not already running
    if ray and not ray.is_initialized():
        ray.init(num_cpus=2, namespace="test", ignore_reinit_error=True)
    yield
    if ray:
        ray.shutdown()


@pytest.mark.skipif(not WEHOOP_BASE.exists(), reason="WEHOOP data repo not available")
def test_ingest_has_rows():
    if load_wehoop is None:
        pytest.skip("ray not installed")

    ds_lines, ds_results = load_wehoop([2023])  # type: ignore[misc]

    assert ds_lines.count() > 0, "Lines dataset empty for 2023 season"
    assert ds_results.count() > 0, "Results dataset empty for 2023 season"