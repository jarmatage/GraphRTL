"""GraphRTL train CLI definition."""

import sys
from pathlib import Path

import click

from graphrtl.yosys import generate_sog_verilog

from .cli import cli


@cli.command()
@click.argument("design", nargs=1, type=str, required=True)
@click.argument(
    "verilog_files",
    nargs=-1,
    type=click.Path(
        dir_okay=False,
        resolve_path=True,
        path_type=Path,
        exists=True,
    ),
    required=True,
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(
        exists=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
    default=Path(),
    help="Output directory for the SOG verilog.",
)
@click.option(
    "--yosys",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=Path,
        executable=True,
    ),
    required=False,
    default=Path("/usr/bin/yosys"),
    help="Path to the yosys executable.",
)
def yosys(
    design: str,
    verilog_files: list[Path],
    outdir: Path,
    yosys: Path,
) -> None:
    """Run yosys on a verilog design to generate an SOG version."""
    generate_sog_verilog(design, verilog_files, outdir, yosys)
    sys.exit(0)
