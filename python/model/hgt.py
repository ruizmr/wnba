"""Minimal Heterogeneous Graph Transformer (HGT) implementation.

This module purposefully keeps the model extremely small so that CPU training
completes in < 10 min per the Week-1 Definition of Done. GPU scaling will be
handled by Agent 2 through Ray Train.

Reference: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020).
"""

from __future__ import annotations

from typing import Dict

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
    from torch_geometric.nn import HGTConv  # type: ignore
    from torch_geometric.data import HeteroData  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    missing = exc.name
    raise ModuleNotFoundError(
        f"{missing} is required for the MiniHGT model. Install via `pip install {missing}` or activate the Conda env "
        "defined in env.yml."
    ) from exc

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class MiniHGT(nn.Module):
    """A tiny 2-layer HGT suitable for smoke tests and CI."""

    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=2,
                    group="sum",
                )
            )

        # Final linear for game node classification (win/lose)
        self.lin_game = nn.Linear(hidden_channels, 2)

    def forward(self, x_dict, edge_index_dict):  # noqa: D401, ANN001
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin_game(x_dict["game"])

    # Convenience -----------------------------------------------------------------

    def infer(self, data: HeteroData):  # noqa: D401
        """Run the network in *eval* mode and return logits for *game* nodes."""

        self.eval()
        with torch.no_grad():
            out = self(data.x_dict, data.edge_index_dict)
        return out