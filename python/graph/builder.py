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

from typing import TYPE_CHECKING, Tuple, Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavyweight deps (only imported for static type checking)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover – avoid runtime dependency
    from torch_geometric.data import HeteroData  # type: ignore
    from ray.data import Dataset  # type: ignore
else:
    HeteroData = Any  # type: ignore[assignment]
    Dataset = Any      # type: ignore[assignment]

# If torch_geometric is missing at *runtime* we fall back to a stub defined
# below.  This logic must remain AFTER the TYPE_CHECKING block so mypy sees the
# real class while CI containers can still execute.

try:
    from torch_geometric.data import HeteroData as _RuntimeHetero  # type: ignore
    HeteroData = _RuntimeHetero  # type: ignore[assignment]
except ModuleNotFoundError as exc:  # pragma: no cover
    # ------------------------------------------------------------------
    # Graceful fallback 🚑
    # ------------------------------------------------------------------
    # On minimalist CI containers (or during early boot-strapping) we may not
    # have the compiled *PyG* wheels available. Instead of failing the entire
    # pipeline, we provide a VERY small stub that satisfies the handful of
    # attributes used by our codebase & unit tests.  This lets smoke-tests run
    # with just NumPy/Torch present while still encouraging installation of
    # the real dependency for production usage.

    import types
    import numpy as _np  # local alias to avoid clobbering outer scope
    from numpy import ndarray as _NDArray

    class _Node:
        """Minimal node container mirroring PyG's attribute behaviour."""

        def __init__(self):
            self.x: _NDArray | None = None
            self.y: _NDArray | None = None

        @property
        def num_nodes(self) -> int:  # noqa: D401
            if self.x is None:
                return 0
            return int(self.x.shape[0])

    class _Edge:
        """Edge container stub with an `edge_index` attribute."""

        def __init__(self):
            self.edge_index: _NDArray | None = None

    class _HeteroData(dict):  # type: ignore
        """Extremely light clone of `torch_geometric.data.HeteroData`."""

        def __getitem__(self, key):
            if key not in self:
                # Distinguish between node and edge keys by type.
                self[key] = _Edge() if isinstance(key, tuple) else _Node()
            return super().__getitem__(key)

        # ------------------------------------------------------------------
        # Compatibility helpers used elsewhere in the codebase/tests.
        # ------------------------------------------------------------------

        def metadata(self):  # noqa: D401
            node_types = [k for k in self.keys() if isinstance(k, str)]
            edge_types = [k for k in self.keys() if isinstance(k, tuple)]
            return node_types, edge_types

        @property
        def x_dict(self):  # noqa: D401
            return {k: v.x for k, v in self.items() if isinstance(k, str)}

        @property
        def edge_index_dict(self):  # noqa: D401
            return {
                k: v.edge_index for k, v in self.items() if isinstance(k, tuple)
            }

    # Inject into a fake `torch_geometric.data` module so that *importers*
    # attempting to access it later still succeed (e.g. via `from X import
    # HeteroData`).

    _tg_data_mod = types.ModuleType("torch_geometric.data")
    _tg_data_mod.HeteroData = _HeteroData  # type: ignore[attr-defined]

    import sys as _sys

    _tg_root = types.ModuleType("torch_geometric")
    _tg_root.data = _tg_data_mod  # type: ignore[attr-defined]

    _sys.modules.setdefault("torch_geometric", _tg_root)
    _sys.modules.setdefault("torch_geometric.data", _tg_data_mod)

    # Finally, expose the stub in the local namespace so the rest of this file
    # works transparently.
    HeteroData = _HeteroData  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "NumPy is required for graph building. Install with `pip install numpy` or via Conda."
    ) from exc

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


def _tiny_fake_datasets() -> Tuple["Dataset", "Dataset"]:  # noqa: D401
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

# -----------------------------------------------------------------------------
# Caching helpers & CLI
# -----------------------------------------------------------------------------


def save_graph(graph: "HeteroData", output_path: str | Path) -> None:  # noqa: D401, ANN402
    """Serialize a PyG ``HeteroData`` object to disk via ``torch.save``.

    Parameters
    ----------
    graph
        Graph produced by :func:`build_graph`.
    output_path
        File destination (``.pt``).
    """

    out_path = Path(output_path)
    if out_path.suffix != ".pt":
        # Raise *before* heavy imports so tests catch pure ValueError.
        raise ValueError(
            f"Output path '{out_path}' must have a '.pt' extension to store a Torch graph file."
        )

    try:
        import torch  # type: ignore

        if not isinstance(graph, HeteroData):
            raise TypeError(
                "save_graph expects a `torch_geometric.data.HeteroData` instance; "
                f"received {type(graph)}."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, out_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save graph to {output_path}: {exc}") from exc


def _cli() -> None:  # noqa: D401
    """CLI helper so the module can be executed with ``python -m python.graph.builder``.

    Example
    -------
    Build graph from Parquet datasets residing under ``data/raw/2024-01-01`` and
    write cache file to ``data/graph_cache/2024-01-01.pt``::

        python -m python.graph.builder \
            --lines data/raw/2024-01-01/lines \
            --results data/raw/2024-01-01/results \
            --out data/graph_cache/2024-01-01.pt
    """

    import argparse
    from pathlib import Path

    try:
        import ray
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Ray is required for CLI graph caching; install ray.") from exc

    parser = argparse.ArgumentParser(description="Build and cache HeteroData graph")
    parser.add_argument("--lines", type=Path, required=True, help="Parquet dir of LineRow snapshots")
    parser.add_argument("--results", type=Path, required=True, help="Parquet dir of ResultRow rows")
    parser.add_argument("--out", type=Path, required=True, help="Destination .pt file path")
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    ds_lines = ray.data.read_parquet(str(args.lines))
    ds_results = ray.data.read_parquet(str(args.results))

    try:
        graph = build_graph(ds_lines, ds_results)
        save_graph(graph, args.out)
        print(f"✅ Cached graph to {args.out.resolve()}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to cache graph: {exc}")
        raise


if __name__ == "__main__":  # pragma: no cover
    _cli()