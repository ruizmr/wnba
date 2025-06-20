"""Command-line interface for the Edge Engine consumer tools.

This file intentionally carries *no* heavy logic—everything complicated lives
in importable modules so that unit tests can exercise it directly. The CLI is
thin sugar for human operators.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.table import Table

from .aggregate import geo_mean
from .lenses import kelly_criterion

app = typer.Typer(help="Edge Engine CLI")
console = Console()

PREDICT_URL_ENV = "PREDICT_URL"


def _mock_prediction() -> Dict[str, Any]:
    """Return a hard-coded mocked prediction used in `--test` mode."""
    return {
        "prob": 0.55,  # win probability
        "odds": 2.2,  # decimal odds
    }


def _fetch_prediction(game_id: str, predict_url: str) -> Dict[str, Any]:
    """Placeholder HTTP request to Serve endpoint (to be wired by Agent 2)."""
    raise NotImplementedError("Live endpoint not yet wired—use --test mode for now.")


@app.command("game")
def predict_game(
    game_id: str = typer.Argument(..., help="ID of the game to predict"),
    test: bool = typer.Option(False, "--test", help="Use mocked Serve response"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Skip writing ledger entries (no side-effects)"
    ),
):
    """Predict a single game and print a Rich table."""
    if test:
        data = _mock_prediction()
    else:
        predict_url = os.getenv(PREDICT_URL_ENV)
        if not predict_url:
            console.print(
                f"[bold red]Environment variable {PREDICT_URL_ENV} not set. "
                "Cannot call Serve endpoint.[/bold red]"
            )
            raise typer.Exit(code=1)
        data = _fetch_prediction(game_id, predict_url)

    stake = kelly_criterion(prob=data["prob"], odds=data["odds"])  # simple lens chain

    table = Table(title="Prediction", box=None)
    table.add_column("Game ID", style="cyan")
    table.add_column("Prob", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("Stake", justify="right", style="green")
    table.add_row(
        game_id,
        f"{data['prob']:.2%}",
        f"{data['odds']}",
        f"{stake:.2%}",
    )
    console.print(table)

    if not dry_run:
        # TODO: call ledger append helper once implemented
        pass


@app.command("slate")
def predict_slate(
    slate_path: Path = typer.Argument(
        ..., exists=True, help="JSON file containing a list of game IDs"
    ),
    test: bool = typer.Option(False, "--test", help="Use mocked Serve responses"),
):
    """Predict a slate (list) of games passed in via JSON filepath."""
    try:
        game_ids: List[str] = json.loads(slate_path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse slate file: {e}[/red]")
        raise typer.Exit(code=1)

    stakes: List[float] = []
    for gid in game_ids:
        if test:
            data = _mock_prediction()
        else:
            predict_url = os.getenv(PREDICT_URL_ENV)
            data = _fetch_prediction(gid, predict_url)
        stake = kelly_criterion(prob=data["prob"], odds=data["odds"])
        stakes.append(stake)

    aggregate_stake = geo_mean(stakes)
    console.print(
        f"[bold]Slate size:[/bold] {len(game_ids)} games • "
        f"[bold green]Aggregate stake[/bold green]: {aggregate_stake:.2%}"
    )


if __name__ == "__main__":
    app()