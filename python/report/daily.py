"""Daily HTML report rendering.

At the moment this is a super-minimal placeholder using an inline Jinja2
template. Once Agent 1 & 2 supply richer metadata we can extend the template
and move it to a standalone file.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

try:
    from jinja2 import Environment, PackageLoader, select_autoescape
except ImportError:  # pragma: no cover
    # Jinja2 may not yet be present in the environment; fall back to f-string.
    Environment = None  # type: ignore


def render_daily_report(predictions: List[Dict[str, Any]], report_date: date | None = None) -> str:
    """Return HTML for the daily edge report.

    Parameters
    ----------
    predictions
        List of prediction dictionaries; the exact schema is TBD.
    report_date
        Date to insert into the report header. Defaults to *today*.
    """
    report_date = report_date or date.today()

    if Environment is None:
        # Fallback plain HTML.
        rows = "".join(
            f"<tr><td>{p.get('game_id','?')}</td><td>{p.get('prob',0):.2%}</td><td>{p.get('odds',0)}</td></tr>"
            for p in predictions
        )
        return f"""<html><body><h1>Daily Edge Report – {report_date}</h1><table>{rows}</table></body></html>"""

    env = Environment(autoescape=select_autoescape())
    template = env.from_string(
        """
        <html>
            <head><title>Daily Edge Report – {{ date }}</title></head>
            <body>
                <h1>Daily Edge Report – {{ date }}</h1>
                <table border="1" cellpadding="4" cellspacing="0">
                    <thead>
                        <tr><th>Game</th><th>Prob</th><th>Odds</th></tr>
                    </thead>
                    <tbody>
                    {% for p in predictions %}
                        <tr>
                            <td>{{ p.game_id }}</td>
                            <td>{{ '%.2f%%' % (p.prob*100) }}</td>
                            <td>{{ p.odds }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </body>
        </html>
        """
    )
    return template.render(date=report_date, predictions=predictions)