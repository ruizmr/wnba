from pathlib import Path

import csv
import tempfile

from python.report.daily import compute_bankroll_metrics, generate_daily_report


def _write_ledger(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    rows = [
        {"date": "2025-01-01", "game_id": "G1", "pnl": "50"},
        {"date": "2025-01-02", "game_id": "G2", "pnl": "-20"},
        {"date": "2025-01-02", "game_id": "G3", "pnl": "30"},
    ]
    with ledger.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return ledger


def test_compute_bankroll_metrics(tmp_path):
    ledger_path = _write_ledger(tmp_path)
    rows = list(csv.DictReader(ledger_path.open()))
    dates, balances, sharpe, sortino = compute_bankroll_metrics(rows, starting_bankroll=1000)
    assert balances[-1] == 1060  # 1000 +50 -20 +30
    assert len(dates) == 2
    # Sharpe/Sortino should be finite numbers
    assert sharpe != 0 or sortino != 0


def test_generate_daily_report(tmp_path):
    ledger_path = _write_ledger(tmp_path)
    report_path = generate_daily_report(ledger_path=ledger_path, output_dir=tmp_path, starting_bankroll=1000)
    assert report_path.exists()
    html = report_path.read_text()
    assert "Daily Edge Report" in html
    assert "Bankroll Curve" in html