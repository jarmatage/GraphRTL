"""Setup the GraphRTL command line interface."""

from . import cli, train, yosys

__all__: list[str] = ["cli", "train", "yosys"]
