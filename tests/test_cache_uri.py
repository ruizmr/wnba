from pathlib import Path

import pytest  # type: ignore

# Skip entire module if core deps missing
try:
    import torch  # type: ignore
except ModuleNotFoundError:
    pytest.skip("torch not installed – skip cache/URI tests", allow_module_level=True)

from python.graph.builder import save_graph, _tiny_fake_datasets, build_graph
from python.model.train import main as train_main


def test_save_graph_extension_validation(tmp_path: Path):
    ds_lines, ds_results = _tiny_fake_datasets()
    graph = build_graph(ds_lines, ds_results)

    # Passing a non-.pt extension should raise ValueError
    bad_path = tmp_path / "graph.bin"
    with pytest.raises(ValueError):
        save_graph(graph, bad_path)


def test_model_uri_file(tmp_path: Path, monkeypatch):
    """Run train.main() in smoke-test mode and ensure latest_uri.txt created."""

    # Redirect models directory to tmp_path to avoid clutter
    monkeypatch.setattr("python.model.train.Path", lambda p="models": tmp_path)
    monkeypatch.setenv("MODEL_URI_PREFIX", "file://")

    # Run smoke-test via train_main (single process, not Ray cluster)
    # We simulate args: --smoke-test --epochs 1
    import sys
    prev = sys.argv.copy()
    sys.argv = ["train.py", "--smoke-test", "--epochs", "1"]
    try:
        train_main()
    finally:
        sys.argv = prev

    uri_file = tmp_path / "latest_uri.txt"
    assert uri_file.exists(), "latest_uri.txt not written"
    content = uri_file.read_text().strip()
    assert content.startswith("file://"), "Model URI does not start with file://"