"""Command-line interface for the GraphRTL package."""

import click
import daiquiri

from graphrtl import __version__


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="info",
    show_default=True,
    help="Set the logging level.",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str = "info") -> None:
    """CLI command group for GraphRTL."""
    daiquiri.setup(level=log_level, program_name="GraphRTL")
    logger = daiquiri.getLogger(__name__)
    ctx.ensure_object(dict)
    logger.info("GraphRTL %s", __version__)
