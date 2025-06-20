from __future__ import annotations

import base64
import csv
import io
import math
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

LEDGER_PATH = Path("ledger.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

__all__ = ["generate_daily_report", "compute_bankroll_metrics"]


# ---------------------------------------------------------------------------
# Core calculation helpers
# ---------------------------------------------------------------------------

def _parse_ledger(path: Path = LEDGER_PATH) -> List[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError("ledger.csv not found – cannot build report.")
    with path.open() as fp:
        rows = list(csv.DictReader(fp))
    return rows


def compute_bankroll_metrics(
    rows: List[dict[str, str]],
    starting_bankroll: float = 1000.0,
) -> Tuple[List[date], List[float], float, float]:
    """Return bankroll curve and Sharpe/Sortino ratios.

    Parameters
    ----------
    rows : list of dict
        Ledger rows sorted by date/time (as loaded from CSV).
    starting_bankroll : float, default 1000.

    Returns
    -------
    dates : list[date]
    balances : list[float]
    sharpe : float
    sortino : float
    """
    # Sort rows by date ascending
    rows_sorted = sorted(rows, key=lambda r: (r["date"], r.get("game_id", "")))

    bankroll = starting_bankroll
    dates: List[date] = []
    balances: List[float] = []
    returns: List[float] = []

    current_day = None
    daily_start = bankroll

    for r in rows_sorted:
        row_date = datetime.fromisoformat(r["date"]).date()
        pnl_str = r.get("pnl", "0")
        try:
            pnl = float(pnl_str)
        except ValueError:
            pnl = 0.0
        bankroll += pnl

        if current_day is None:
            current_day = row_date
        if row_date != current_day:
            # record end-of-day balance and daily return
            daily_end = bankroll
            returns.append((daily_end - daily_start) / daily_start if daily_start else 0.0)
            dates.append(current_day)
            balances.append(daily_end)
            # reset for new day
            current_day = row_date
            daily_start = bankroll

    # Record last day
    if current_day is not None:
        daily_end = bankroll
        returns.append((daily_end - daily_start) / daily_start if daily_start else 0.0)
        dates.append(current_day)
        balances.append(daily_end)

    # Compute Sharpe/Sortino (annualised, 252 trading days)
    if len(returns) < 2 or all(r == 0 for r in returns):
        sharpe = sortino = 0.0
    else:
        mean_ret = sum(returns) / len(returns)
        std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1))
        downside_std = math.sqrt(
            sum((min(0, r)) ** 2 for r in returns) / max(1, sum(r < 0 for r in returns))
        )
        annual_factor = math.sqrt(252)
        sharpe = mean_ret / std_ret * annual_factor if std_ret else 0.0
        sortino = mean_ret / downside_std * annual_factor if downside_std else 0.0

    return dates, balances, sharpe, sortino


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_bankroll(dates: List[date], balances: List[float]) -> str:
    """Return base64-encoded PNG of bankroll curve."""
    if not dates:
        return ""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(dates, balances, marker="o", color="#1f77b4")
    ax.set_title("Bankroll Over Time")
    ax.set_ylabel("Balance")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_daily_report(
    *,
    ledger_path: Path = LEDGER_PATH,
    output_dir: Path = REPORTS_DIR,
    starting_bankroll: float = 1000.0,
) -> Path:
    """Generate HTML daily report and return file path."""
    rows = _parse_ledger(ledger_path)
    dates, balances, sharpe, sortino = compute_bankroll_metrics(rows, starting_bankroll)

    img_b64 = _plot_bankroll(dates, balances)
    today_str = date.today().isoformat()
    outfile = output_dir / f"daily_report_{today_str}.html"

    # Very small inline HTML, could be replaced by Jinja2
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<meta charset=\"utf-8\">
<title>Daily Edge Report – {today_str}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
header {{ margin-bottom: 20px; }}
h1 {{ color: #333; }}
.metric-card {{ display: inline-block; padding: 10px 20px; margin-right: 20px; background:#f5f5f5; border-radius: 6px; }}
.metric-card h2 {{ margin: 0 0 5px 0; font-size: 1.1rem; }}
.metric-card p {{ margin: 0; font-size: 1.2rem; font-weight: bold; }}
</style>
<body>
<header>
  <h1>Daily Edge Report – {today_str}</h1>
  <div class=\"metric-card\"><h2>Sharpe</h2><p>{sharpe:.2f}</p></div>
  <div class=\"metric-card\"><h2>Sortino</h2><p>{sortino:.2f}</p></div>
</header>
<section>
  <h2>Bankroll Curve</h2>
  {f'<img src="data:image/png;base64,{img_b64}" alt="bankroll chart">' if img_b64 else '<p>No data available.</p>'}
</section>
</body>
</html>"""
    outfile.write_text(html)
    return outfile