import pytest  # type: ignore

import torch  # type: ignore

from python.data.nightly_fetch import _fake_lines, _fake_results
from python.graph.builder import build_graph
from python.model.hgt import MiniHGT


def test_forward_pass_cpu():
    """Minimal smoke test that runs a forward pass on CPU (no CUDA required)."""

    from datetime import date
    import ray

    ray.init(ignore_reinit_error=True, address="local")
    ds_lines = ray.data.from_items(_fake_lines(10, date.today()))
    ds_results = ray.data.from_items(_fake_results(5, date.today()))
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