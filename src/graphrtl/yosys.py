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


def build_yosys_script(
    design: str, verilog_files: list[Path], output_verilog: Path = Path()
) -> str:
    """Build a Yosys script to generate SOG verilog."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ys") as file:
        for verilog_file in verilog_files:
            file.write(f"read_verilog {verilog_file}\n")
        file.write(f"hierarchy -check -top {design}\n")
        file.write("\n".join(SOG_YOSYS_SCRIPT))
        file.write(f"\nwrite_verilog {output_verilog}\n")
        return file.name


def clean_verilog(input_verilog: Path, output_verilog: Path) -> None:
    """Clean a verilog file by removing specific annotations and empty lines."""
    with input_verilog.open("r") as fin, output_verilog.open("w") as fout:
        for line in fin:
            processed: str = re.sub(RE_REMOVE, "", line)
            if processed.strip():
                fout.write(processed)


def generate_sog_verilog(
    design: str,
    verilog_files: list[Path],
    outdir: Path = Path(),
    yosys: Path = Path("/usr/bin/yosys"),
) -> None:
    """Convert a set of verilog files into a single SOG verilog file using Yosys."""
    tmp_sog: Path = outdir / "tmp.sog.v"
    script: str = build_yosys_script(design, verilog_files, tmp_sog)
    subprocess.run([str(yosys), script], check=True)  # noqa: S603
    clean_sog: Path = outdir / f"{design}.sog.v"
    clean_verilog(tmp_sog, clean_sog)
    tmp_sog.unlink()
    Path(script).unlink()
