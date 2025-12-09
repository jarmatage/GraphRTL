"""Module to generate SOG verilog using Yosys."""

import re
import subprocess
import tempfile
from pathlib import Path

SOG_YOSYS_SCRIPT: list[str] = [
    "proc",
    "flatten",
    "opt",
    "fsm",
    "opt",
    "memory",
    "opt",
    "techmap",
    "opt",
]

RE_REMOVE = r"\(\*(.*)\*\)"


def generate_sog_verilog(
    design: str,
    verilog_files: list[Path],
    outdir: Path = Path(),
    yosys: Path = Path("/usr/bin/yosys"),
) -> None:
    """Convert a set of verilog files into a single SOG verilog file using Yosys."""
    tmp_sog: Path = outdir / "tmp.sog.v"
    clean_sog: Path = outdir / f"{design}.sog.v"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ys") as file:
        for verilog_file in verilog_files:
            file.write(f"read_verilog {verilog_file}\n")
        file.write(f"hierarchy -check -top {design}\n")
        file.write("\n".join(SOG_YOSYS_SCRIPT))
        file.write(f"\nwrite_verilog {tmp_sog}\n")
        script: str = file.name

    subprocess.run([str(yosys), script], check=True)  # noqa: S603

    with tmp_sog.open("r") as fin, clean_sog.open("w") as fout:
        for line in fin:
            if line.strip():
                fout.write(re.sub(RE_REMOVE, "", line))
