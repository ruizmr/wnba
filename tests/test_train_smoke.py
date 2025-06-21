import pytest  # type: ignore

# Ensure core deps present; otherwise skip whole module
try:
    import ray  # type: ignore
except ModuleNotFoundError:
    pytest.skip("ray not installed – skip smoke tests", allow_module_level=True)

import torch  # type: ignore

from python.graph.builder import build_graph, _tiny_fake_datasets
from python.model.hgt import MiniHGT


def test_forward_pass_cpu():
    """Minimal smoke test that runs a forward pass on CPU (no CUDA required)."""

    from datetime import date

    ray.init(ignore_reinit_error=True, address="local")
    ds_lines, ds_results = _tiny_fake_datasets()
    g = build_graph(ds_lines, ds_results)

    model = MiniHGT(metadata=g.metadata(), hidden_channels=32, num_layers=1)
    logits = model(g.x_dict, g.edge_index_dict)
    assert logits.shape[0] == g["game"].num_nodes
    assert logits.shape[1] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_warning():
    """Ensure CUDA path at least initialises without crash if GPU present."""

    # Quick check that torch can allocate a tensor on CUDA
    tensor = torch.randn(1).to("cuda")
    assert tensor.device.type == "cuda"