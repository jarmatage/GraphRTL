"""Unit tests for the CLI module."""

from click.testing import CliRunner

from graphrtl import __version__
from graphrtl.cli.cli import cli


def test_cli_help() -> None:
    """Test CLI help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CLI command group for GraphRTL" in result.output


def test_cli_version_option() -> None:
    """Test CLI --version option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_log_level_debug() -> None:
    """Test CLI with DEBUG log level."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--log-level", "DEBUG", "train"])
    assert result.exit_code == 0
