"""Graph builder that converts tabular datasets into a heterogeneous graph.

This simplified implementation uses :pyclass:`networkx.MultiDiGraph` so that
unit tests do not depend on `torch_geometric`.  The structure mirrors the
planned schema sufficiently for downstream model prototyping.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Sequence

# Soft dependency: networkx
try:
    import networkx as nx  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("networkx must be installed to use graph.builder") from exc

__all__ = ["build_graph"]


class GameLineRow(dict):
    """Type alias standing in for a real dataclass/ORM row."""

    game_id: str
    line: float  # home – away spread (closing line)
    timestamp_utc: datetime


class ResultRow(dict):
    game_id: str
    home_score: int
    away_score: int


class PbpEventRow(dict):
    game_id: str
    event_id: int
    # ... other keys not needed for line edges


# Edge labels (constants) ----------------------------------------------------
GAME_HAS_LINE = "game-has-line"
LINE_MOVE = "line-move"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    ds_lines: Sequence[GameLineRow],
    ds_results: Sequence[ResultRow] | None = None,
    ds_pbp: Sequence[PbpEventRow] | None = None,
):
    """Convert datasets into a heterogeneous MultiDiGraph.

    Parameters
    ----------
    ds_lines
        Iterable of records with at least ``game_id``, ``timestamp_utc`` and
        ``line`` fields.
    ds_results, ds_pbp
        Currently unused but kept for signature compatibility.
    """
    g = nx.MultiDiGraph()

    # ---------------------------------------------------------------------
    # Add Game nodes (one per unique game_id)
    # ---------------------------------------------------------------------
    for row in ds_lines:
        game_id = row["game_id"]
        if not g.has_node(game_id):
            g.add_node(game_id, node_type="game", game_id=game_id)

    # ---------------------------------------------------------------------
    # Add VegasLine nodes and (game-has-line) edges
    # ---------------------------------------------------------------------
    # We'll group lines per game to add line-move edges in chronological order
    lines_by_game: dict[str, List[GameLineRow]] = {}
    for row in ds_lines:
        lines_by_game.setdefault(row["game_id"], []).append(row)

    for game_id, rows in lines_by_game.items():
        # Sort by timestamp for deterministic ordering
        rows.sort(key=lambda r: r["timestamp_utc"])
        prev_line_node = None
        for idx, row in enumerate(rows):
            line_id = f"line::{game_id}::{idx}"
            g.add_node(
                line_id,
                node_type="vegas_line",
                game_id=game_id,
                line=row["line"],
                timestamp=row["timestamp_utc"],
            )
            # Edge Game -> VegasLine
            g.add_edge(game_id, line_id, key=GAME_HAS_LINE, edge_type=GAME_HAS_LINE)

            # Edge VegasLine[t-1] -> VegasLine[t]
            if prev_line_node is not None:
                g.add_edge(
                    prev_line_node,
                    line_id,
                    key=LINE_MOVE,
                    edge_type=LINE_MOVE,
                )
            prev_line_node = line_id

    return g