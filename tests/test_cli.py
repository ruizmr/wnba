"""Integration-ish tests for CLI entrypoints (mocked mode only)."""
from typer.testing import CliRunner

from python.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_predict_game_test():
    result = runner.invoke(app, ["game", "GAME123", "--test", "--dry-run"])
    assert result.exit_code == 0
    assert "GAME123" in result.output