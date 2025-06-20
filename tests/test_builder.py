import pytest  # type: ignore

from python.graph.builder import _tiny_fake_datasets, build_graph


def test_build_graph_sanity():
    ds_lines, ds_results = _tiny_fake_datasets()
    g = build_graph(ds_lines, ds_results)
    # Expect at least one node of each type and matching edge counts.
    assert g["team"].num_nodes >= 2
    assert g["game"].num_nodes >= 1
    assert g["team", "participates", "game"].edge_index.shape[1] >= 2  # at least 2 edges