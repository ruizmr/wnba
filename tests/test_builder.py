"""Unit tests for graph.builder with Vegas line edges."""

from datetime import datetime, timedelta

import pytest

# Skip module if networkx missing
pytest.importorskip("networkx")
import networkx as nx  # noqa: E402

from python.graph.builder import GAME_HAS_LINE, LINE_MOVE, build_graph


def _fake_line_rows(game_id: str, n: int = 3):
    base_ts = datetime(2024, 6, 1, 12, 0, 0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "game_id": game_id,
                "line": -3.5 + i,  # just vary a bit
                "timestamp_utc": base_ts + timedelta(minutes=i * 10),
            }
        )
    return rows


def test_line_edges_and_nodes():
    # create synthetic snapshot lines for two games
    ds_lines = _fake_line_rows("g1", 4) + _fake_line_rows("g2", 2)

    g = build_graph(ds_lines)

    # Each game node should exist
    for gid in ["g1", "g2"]:
        assert g.nodes[gid]["node_type"] == "game"

        # count outgoing game-has-line edges
        edges = [e for _, t, k, d in g.out_edges(gid, keys=True, data=True) if d["edge_type"] == GAME_HAS_LINE]
        assert len(edges) == (4 if gid == "g1" else 2)

    # Validate line-move edges count and direction ordering for g1
    lines_nodes = [n for n, d in g.nodes(data=True) if d["node_type"] == "vegas_line" and d["game_id"] == "g1"]
    # Expect 4 nodes hence 3 line-move edges
    line_move_edges = [edge for edge in g.edges(keys=True, data=True) if edge[3]["edge_type"] == LINE_MOVE and g.nodes[edge[0]]["game_id"] == "g1"]
    assert len(line_move_edges) == 3

    # Ensure edges follow correct order (source timestamp < target timestamp)
    for src, tgt, _, _ in line_move_edges:
        assert g.nodes[src]["timestamp"] < g.nodes[tgt]["timestamp"]