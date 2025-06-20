"""Top-level package for Application Dev & QA components.

This package will contain the lenses, aggregation logic, CLI, and report generation
utilities that sit on top of the prediction service provided by other agents.

Example
-------
>>> from python.lenses import kelly_criterion
>>> kelly_criterion(edge=0.05, bankroll_fraction=0.01)
0.0005
"""

__all__ = [
    "lenses",
    "aggregate",
    "cli",
    "report",
]