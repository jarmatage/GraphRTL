"""GraphRTL train CLI definition."""

import sys

import click

from .cli import cli


@cli.command()
def train() -> None:
    """Train a GraphRTL model."""
    click.echo("Training GraphRTL model...")
    sys.exit(0)
