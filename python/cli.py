from __future__ import annotations

import datetime as _dt
import csv
import os
import random
from pathlib import Path
from typing import List

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "typer is required for the CLI. Install via `pip install --break-system-packages typer click rich`."
    ) from exc

from rich.console import Console
from rich.table import Table

from python.lenses import kelly_fraction
from python.aggregate import geo_mean

app = typer.Typer(add_completion=False)
console = Console()

LEDGER_PATH = Path("ledger.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_ledger(row: dict[str, str], dry_run: bool = False) -> None:
    """Append a *row* to ledger.csv unless *dry_run* is True."""
    if dry_run:
        console.print("[yellow]Dry-run: not writing to ledger.csv[/yellow]")
        return

    exists = LEDGER_PATH.exists()
    with LEDGER_PATH.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def predict_game(
    game_id: str = typer.Argument(..., help="Unique game identifier"),
    win_prob: float = typer.Option(..., help="Model win probability (0-1)"),
    decimal_odds: float = typer.Option(..., help="Sportsbook decimal odds (>1.0)"),
    bankroll: float = typer.Option(1000.0, help="Current bankroll size"),
    multiplier: float = typer.Option(1.0, help="Kelly multiplier (e.g. 0.5 = half)"),
    dry_run: bool = typer.Option(False, help="Do not write ledger row"),
):
    """Predict a single game, returning stake suggestions and optional ledger entry."""
    stake_frac = kelly_fraction(win_prob, decimal_odds, multiplier=multiplier)
    stake_size = bankroll * stake_frac

    table = Table(title=f"Prediction for game {game_id}")
    table.add_column("Win Prob", justify="right")
    table.add_column("Dec Odds", justify="right")
    table.add_column("Kelly %", justify="right")
    table.add_column("Stake", justify="right")
    table.add_row(f"{win_prob:.3f}", f"{decimal_odds:.2f}", f"{stake_frac:.2%}", f"${stake_size:.2f}")
    console.print(table)

    row = {
        "date": _dt.date.today().isoformat(),
        "game_id": game_id,
        "stake_frac": f"{stake_frac:.6f}",
        "stake_size": f"{stake_size:.2f}",
        "win_prob": f"{win_prob:.4f}",
        "decimal_odds": f"{decimal_odds:.2f}",
        "kelly_multiplier": f"{multiplier}",
        "pnl": "NA",
    }
    _append_ledger(row, dry_run=dry_run)


@app.command()
def predict_ledger(
    slate_date: str = typer.Option(None, help="Date YYYY-MM-DD; defaults to today"),
    n_games: int = typer.Option(5, help="Number of games to mock"),
    bankroll: float = typer.Option(1000.0, help="Current bankroll"),
    multiplier: float = typer.Option(1.0, help="Kelly multiplier"),
    dry_run: bool = typer.Option(False, help="Do not write ledger rows"),
):
    """Mock slate prediction → pretty table & ledger rows.

    In real usage this would call the Ray Serve endpoint; here we randomly
    sample win probabilities and odds for demonstration / smoke-test purposes.
    """
    date = _dt.date.fromisoformat(slate_date) if slate_date else _dt.date.today()

    game_ids: List[str] = [f"GAME-{date:%Y%m%d}-{i+1}" for i in range(n_games)]
    # Fake predictions
    win_probs = [round(random.uniform(0.4, 0.65), 3) for _ in range(n_games)]
    decimal_odds = [round(random.uniform(1.8, 2.2), 2) for _ in range(n_games)]

    table = Table(title=f"Slate prediction for {date.isoformat()}")
    table.add_column("Game ID")
    table.add_column("WinProb", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("Kelly%", justify="right")
    table.add_column("Stake", justify="right")

    for gid, p, o in zip(game_ids, win_probs, decimal_odds):
        kelly_pct = kelly_fraction(p, o, multiplier=multiplier)
        stake_size = bankroll * kelly_pct
        table.add_row(gid, f"{p:.3f}", f"{o:.2f}", f"{kelly_pct:.2%}", f"${stake_size:.2f}")

        row = {
            "date": date.isoformat(),
            "game_id": gid,
            "stake_frac": f"{kelly_pct:.6f}",
            "stake_size": f"{stake_size:.2f}",
            "win_prob": f"{p:.4f}",
            "decimal_odds": f"{o:.2f}",
            "kelly_multiplier": f"{multiplier}",
            "pnl": "NA",
        }
        _append_ledger(row, dry_run=dry_run)

    console.print(table)


@app.command()
def view_ledger(head: int = typer.Option(10, help="Number of rows to show")):
    """Print the last *head* rows of the ledger."""
    if not LEDGER_PATH.exists():
        console.print("[red]ledger.csv not found[/red]")
        raise typer.Exit(code=1)

    rows = list(csv.DictReader(LEDGER_PATH.open()))
    rows_tail = rows[-head:]

    table = Table(title=f"Last {head} ledger rows")
    for col in rows_tail[0].keys():
        table.add_column(col)
    for row in rows_tail:
        table.add_row(*[row[c] for c in table.columns.keys()])
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()