"""Graph builder utilities.

The *graph* is the beating heart of the predictive engine – a heterogeneous
network that encodes relationships between *teams*, *games*, and *lines*.

For the MVP we keep things stupid‐simple:
1. Each **team** is a node with features = one‐hot (n_teams).
2. Each **game** is a node with features = \[home_one_hot, away_one_hot].
3. Edges: team → game participation.

The builder intentionally stays stateless & pure so that it can run inside a Ray
remote task *or* locally for unit tests.
"""

from __future__ import annotations

from typing import Tuple

try:
    from torch_geometric.data import HeteroData  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "torch-geometric is required for graph building. Add it to env.yml or install manually."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("numpy is required. Please install numpy.") from exc

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_graph(ds_lines, ds_results) -> HeteroData:  # noqa: D401, ANN001
    """Build a PyG `HeteroData` object from Ray datasets.

    Parameters
    ----------
    ds_lines
        `ray.data.Dataset` containing `LineRow` dicts.
    ds_results
        `ray.data.Dataset` containing `ResultRow` dicts.

    Returns
    -------
    torch_geometric.data.HeteroData
        Graph with ``team`` and ``game`` node types.
    """

    # Collect unique teams + games.
    teams = sorted({row["team"] for row in ds_lines.take_all()})
    team_to_idx = {t: i for i, t in enumerate(teams)}

    games = sorted({row["game_id"] for row in ds_lines.take_all()})
    game_to_idx = {g: i for i, g in enumerate(games)}

    # One‐hot encode team features.
    x_team = np.eye(len(teams), dtype=np.float32)

    # Game features: concatenated home/away one‐hot teams.
    x_game = np.zeros((len(games), len(teams) * 2), dtype=np.float32)
    for row in ds_lines.take_all():
        g_idx = game_to_idx[row["game_id"]]
        t_idx = team_to_idx[row["team"]]
        if row["value"] < 0:  # negative spread -> favorite (home?)
            x_game[g_idx, t_idx] = 1.0  # simplistic placeholder
        else:
            x_game[g_idx, len(teams) + t_idx] = 1.0

    # Edge index arrays.
    src, dst = [], []
    for row in ds_lines.take_all():
        src.append(team_to_idx[row["team"]])
        dst.append(game_to_idx[row["game_id"]])

    edge_index = np.stack([src, dst], axis=0)

    data = HeteroData()
    data["team"].x = np.asarray(x_team)
    data["game"].x = np.asarray(x_game)
    data["team", "participates", "game"].edge_index = edge_index

    # Attach labels (team won?)
    y = np.zeros(len(games), dtype=int)
    for row in ds_results.take_all():
        if row["won"]:
            y[game_to_idx[row["game_id"]]] = 1  # mark winners
    data["game"].y = y

    return data

# -----------------------------------------------------------------------------
# Simple smoke test – executed via `pytest -q`.
# -----------------------------------------------------------------------------


def _tiny_fake_datasets():  # noqa: D401
    """Return small in‐memory Ray datasets for testing."""

    try:
        import ray
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Ray required for graph tests; install ray.") from exc

    ray.init(ignore_reinit_error=True, address="local")

    ds_lines = ray.data.from_items(
        [
            {"game_id": 1, "team": "LAL", "line_type": "spread", "value": -3.5, "odds": -110},
            {"game_id": 1, "team": "NYK", "line_type": "spread", "value": 3.5, "odds": -110},
        ]
    )

    ds_results = ray.data.from_items(
        [
            {"game_id": 1, "team": "LAL", "points": 102, "won": True},
            {"game_id": 1, "team": "NYK", "points": 97, "won": False},
        ]
    )

    return ds_lines, ds_results


def _sanity_check() -> Tuple[int, int]:  # noqa: D401
    ds_lines, ds_results = _tiny_fake_datasets()
    graph = build_graph(ds_lines, ds_results)
    return graph["team"].num_nodes, graph["game"].num_nodes