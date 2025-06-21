"""Graph builder that operates on Ray `Dataset`s in a streaming fashion.

This refactor replaces any `.take_all()` call with a memory-safe batch loop
using ``dataset.iter_batches``.  It ensures we never materialise the full
play-by-play dataset on the driver, which previously caused OOM for large
season ranges.

The public ``build_graph`` entry point returns a very lightweight
``GraphStats`` data-structure that contains the minimal information our unit
and smoke tests rely on (counts of teams / games / edges and optional feature
matrices).  Down-stream modelling code can be extended to consume richer
artifacts without changing the streaming pattern introduced here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

try:
    import ray
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "`ray` must be installed to use the graph builder (pip install ray)."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class GraphStats:
    """A very small footprint container with graph statistics.

    Rather than constructing the *actual* graph structure (adjacency matrices
    etc.) inside the driver process we only keep high-level aggregates that are
    cheap to hold in memory.  The heavy lifting is expected to happen on Ray
    workers or GPU devices later in the pipeline.
    """

    seasons: List[int]
    team_count: int
    game_count: int
    edge_count: int

    # Optional, keyed by feature name → list/array of features per team
    team_features: Dict[str, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_unique(values: pd.Series, dest: Set):  # type: ignore[type-var]
    """Add unique (non-null) values from ``values`` into *dest*."""

    dest.update(v for v in values.unique().tolist() if v is not None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    ds_lines: "ray.data.Dataset",
    ds_results: "ray.data.Dataset",
    *,
    batch_size: int = 8192,
    extra_node_features: Optional[List[str]] = None,
) -> GraphStats:
    """Build season graph in a streaming, memory-safe manner.

    Parameters
    ----------
    ds_lines
        Play-by-play events Ray Dataset (already projected via *wehoop_ingest*).
    ds_results
        Team boxscore Ray Dataset.
    batch_size
        Number of rows per batch when iterating through *ds_lines*.
    extra_node_features
        List of column names (from *ds_results*) to aggregate as **mean** per
        team.  Example: ``["season", "rest_days"]``.  If *None* (default) no
        additional features are calculated.
    """

    extra_node_features = extra_node_features or []

    teams: Set[int] = set()
    games: Set[int] = set()
    edge_count = 0

    logger.info("Iterating over play-by-play batches (batch_size=%d)…", batch_size)

    for batch in ds_lines.iter_batches(batch_size=batch_size, batch_format="pandas"):
        assert isinstance(batch, pd.DataFrame)
        _collect_unique(batch["home_team_id"], teams)  # type: ignore[arg-type]
        _collect_unique(batch["away_team_id"], teams)  # type: ignore[arg-type]
        _collect_unique(batch["game_id"], games)  # type: ignore[arg-type]

        edge_count += len(batch)

    logger.info(
        "Completed aggregation: teams=%d games=%d edges=%d",
        len(teams),
        len(games),
        edge_count,
    )

    # ---------------------------------------------------------------------
    # Aggregate extra node features from the *results* dataset.
    # We keep the aggregation logic simple (mean per team) – this can be
    # replaced with something more sophisticated later.
    # ---------------------------------------------------------------------
    team_features: dict[str, list] = {}
    if extra_node_features:

        def _agg_fn(batch: pd.DataFrame):  # type: ignore[name-defined]
            return (
                batch.groupby("team_id")[extra_node_features]
                .mean(numeric_only=True)
                .reset_index()
            )

        df_features = ds_results.map_batches(_agg_fn, batch_format="pandas").to_pandas()
        for feat in extra_node_features:
            # Map team_id → feature value list (aligning order with *teams* set)
            mapping = dict(zip(df_features["team_id"], df_features[feat]))
            team_features[feat] = [mapping.get(t) for t in teams]

    seasons_sorted = sorted({row["season"] for row in ds_results.take(1000)})

    return GraphStats(
        seasons=seasons_sorted,
        team_count=len(teams),
        game_count=len(games),
        edge_count=edge_count,
        team_features=team_features,
    )