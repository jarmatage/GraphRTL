# How to Gather RTL Designs with Power Labels

This guide explains how to collect training data (RTL designs + power labels) for your GNN power estimation model.

## Overview

To train a power estimation model, you need:
1. **RTL designs** (Verilog/SystemVerilog files)
2. **Power labels** (actual power consumption values)

The power labels typically come from EDA tools after synthesis.

## Method 1: Using Commercial EDA Tools (Recommended)

### Step 1: Collect RTL Designs

**Sources of RTL designs:**
- Your own designs (internal IP)
- Open-source projects (GitHub, OpenCores)
- Academic benchmarks (ISCAS, IWLS)
- Industry standard benchmarks

**Popular open-source RTL repositories:**
```bash
# Example: Clone some open-source designs
git clone https://github.com/chipsalliance/rocket-chip
git clone https://github.com/lowRISC/ibex
git clone https://github.com/ultraembedded/riscv
git clone https://github.com/YosysHQ/picorv32
```

### Step 2: Synthesize and Get Power Labels

You need to synthesize each design and extract power consumption. Here's the workflow:

#### Using Synopsys Design Compiler

```tcl
# synthesis_script.tcl

# Set up libraries
set_app_var search_path "/path/to/libraries"
set_app_var target_library "your_target_lib.db"
set_app_var link_library "* your_target_lib.db"

# Read RTL
read_verilog design.v

# Set constraints
create_clock -period 10 [get_ports clk]
set_input_delay -clock clk 1 [all_inputs]
set_output_delay -clock clk 1 [all_outputs]

# Synthesize
compile_ultra

# Get power report
set_switching_activity -toggle_rate 0.1 -static_probability 0.5 [all_nets]
report_power -analysis_effort high > power_report.txt

# Export data
# Parse power_report.txt to extract total power
```

**Extract power from report:**
```bash
# power_report.txt will contain lines like:
# Total Dynamic Power    = 123.45 mW
# Cell Leakage Power     = 12.34 mW
# Total Power            = 135.79 mW

# Parse and save to JSON
python extract_power.py power_report.txt design_name.json
```

#### Using Cadence Genus

```tcl
# genus_script.tcl

# Read design
read_hdl design.v

# Elaborate
elaborate design_name

# Set constraints
create_clock -period 10 clk

# Synthesize
syn_generic
syn_map
syn_opt

# Power analysis
set_power_analysis_mode -reset
set_power_analysis_mode -method static
set_power_analysis -corner max
report_power > power_report.txt
```

#### Using Open-Source Tools (Yosys + OpenSTA)

For a free alternative:

```bash
# 1. Synthesize with Yosys
yosys -p "
    read_verilog design.v;
    synth -top design_name;
    abc -liberty liberty_file.lib;
    write_verilog design_synth.v;
    stat -liberty liberty_file.lib
" > synth_report.txt

# 2. Get timing info with OpenSTA
sta << EOF
    read_liberty liberty_file.lib
    read_verilog design_synth.v
    link_design design_name
    create_clock -period 10 clk
    report_checks
EOF

# 3. Estimate power (simplified)
# Note: This is approximate - commercial tools are more accurate
python estimate_power_from_netlist.py design_synth.v liberty_file.lib
```

### Step 3: Create Label Files

Create JSON files with power labels:

```json
{
  "Power": 135.79,
  "Area": 1234.56,
  "TNS": -1.23,
  "WNS": -0.45,
  "slack_worst": -0.45,
  "frequency": 100.0,
  "technology": "45nm",
  "voltage": 1.1,
  "timestamp": "2025-12-09"
}
```

## Method 2: Using MasterRTL's Existing Data

If you have access to MasterRTL data, you can use it directly:

```python
# MasterRTL already has labels in this format
import json

# Example: Load TinyRocket power label
with open("MasterRTL/example/label/TinyRocket.json") as f:
    labels = json.load(f)

print(f"Power: {labels['Power']} mW")
print(f"Area: {labels['Area']} um²")
```

## Method 3: Simulation-Based Power Estimation

For smaller designs without access to synthesis tools:

### Using Icarus Verilog + GTKWave

```bash
# 1. Create testbench
# testbench.v
module testbench;
    reg clk, rst;
    // ... instantiate your design
    
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, testbench);
        // ... stimulus
    end
endmodule

# 2. Simulate
iverilog -o sim design.v testbench.v
vvp sim

# 3. Analyze activity
python analyze_activity.py waveform.vcd
# This will estimate switching activity for power calculation
```

### Using Verilator

```bash
# 1. Create C++ testbench
verilator --cc design.v --exe sim_main.cpp

# 2. Build and run
make -C obj_dir -f Vdesign.mk
./obj_dir/Vdesign

# 3. Collect activity statistics
# Parse VCD or use Verilator's --trace option
```

## Method 4: Using Published Benchmarks

Several academic datasets are available:

### ISPD/ICCAD Benchmarks
- Download from contest websites
- Usually include power reports
- Good for validation/testing

### OpenROAD Flow Datasets
```bash
git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd OpenROAD-flow-scripts
# Contains designs with power data
```

## Automation Script

Here's a script to automate data collection:

```python
#!/usr/bin/env python3
"""
Automated RTL design power labeling pipeline.
"""

import json
import subprocess
from pathlib import Path

def synthesize_and_extract_power(verilog_file: Path, output_dir: Path) -> dict:
    """
    Synthesize RTL and extract power consumption.
    
    This is a template - adapt to your EDA tool.
    """
    design_name = verilog_file.stem
    
    # 1. Run synthesis (example with Yosys)
    synth_script = f"""
    read_verilog {verilog_file}
    hierarchy -check -top {design_name}
    synth -top {design_name}
    stat
    """
    
    with open("temp_synth.ys", "w") as f:
        f.write(synth_script)
    
    result = subprocess.run(
        ["yosys", "-s", "temp_synth.ys"],
        capture_output=True,
        text=True
    )
    
    # 2. Parse statistics
    stats = parse_yosys_output(result.stdout)
    
    # 3. Estimate power (simplified)
    power = estimate_power_from_stats(stats)
    
    # 4. Create label file
    label = {
        "Power": power,
        "Area": stats.get("area", 0),
        "num_cells": stats.get("num_cells", 0),
        "num_wires": stats.get("num_wires", 0),
    }
    
    label_file = output_dir / f"{design_name}.json"
    with open(label_file, "w") as f:
        json.dump(label, f, indent=2)
    
    return label

def parse_yosys_output(output: str) -> dict:
    """Parse Yosys statistics output."""
    stats = {}
    for line in output.split("\n"):
        if "Number of cells:" in line:
            stats["num_cells"] = int(line.split()[-1])
        if "Number of wires:" in line:
            stats["num_wires"] = int(line.split()[-1])
        # Add more parsing as needed
    return stats

def estimate_power_from_stats(stats: dict) -> float:
    """
    Simplified power estimation.
    
    For production, use actual synthesis tool power reports.
    """
    # Very simplified model (replace with actual data)
    num_cells = stats.get("num_cells", 0)
    num_wires = stats.get("num_wires", 0)
    
    # Rough estimate: 0.1 mW per cell + 0.01 mW per wire
    power = num_cells * 0.1 + num_wires * 0.01
    
    return round(power, 4)

def main():
    """Process all RTL designs in a directory."""
    verilog_dir = Path("rtl_designs")
    output_dir = Path("labels")
    output_dir.mkdir(exist_ok=True)
    
    verilog_files = list(verilog_dir.glob("*.v")) + list(verilog_dir.glob("*.sv"))
    
    print(f"Found {len(verilog_files)} RTL designs")
    
    for verilog_file in verilog_files:
        print(f"\nProcessing {verilog_file.name}...")
        try:
            label = synthesize_and_extract_power(verilog_file, output_dir)
            print(f"  Power: {label['Power']:.4f} mW")
            print(f"  Saved to {output_dir / verilog_file.stem}.json")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
```

## Data Collection Best Practices

### 1. **Diversity**
Collect designs with:
- Different sizes (small to large)
- Different types (datapaths, control logic, mixed)
- Different architectures (pipelined, combinational, FSMs)

### 2. **Quality**
Ensure:
- Synthesizable RTL (no simulation-only constructs)
- Consistent synthesis settings
- Same technology library
- Same operating conditions (voltage, temperature)

### 3. **Quantity**
Recommended dataset sizes:
- **Minimum**: 50 designs for initial experiments
- **Good**: 100-500 designs for production model
- **Ideal**: 1000+ designs for best accuracy

### 4. **Consistency**
Keep constant:
- Clock frequency (or normalize by frequency)
- Technology node
- Voltage
- Corner (typical, fast, slow)
- Switching activity assumptions

## Example Data Organization

```
project/
├── rtl_designs/
│   ├── design1.v
│   ├── design2.v
│   └── design3.sv
├── labels/
│   ├── design1.json
│   ├── design2.json
│   └── design3.json
└── synthesis_reports/
    ├── design1_power.txt
    ├── design2_power.txt
    └── design3_power.txt
```

## Quick Start with Open-Source Designs

Here's how to quickly build a small dataset:

```bash
#!/bin/bash
# collect_dataset.sh

# 1. Clone open-source designs
mkdir -p rtl_designs
cd rtl_designs

# Small RISC-V core
git clone https://github.com/YosysHQ/picorv32
cp picorv32/picorv32.v ./

# More designs from OpenCores
wget https://opencores.org/websvn/filedetails?repname=...

# 2. For each design, synthesize and get power
cd ..
mkdir -p labels

for design in rtl_designs/*.v; do
    name=$(basename "$design" .v)
    echo "Processing $name..."
    
    # Synthesize (adapt to your tools)
    python synthesize_and_label.py "$design" "labels/${name}.json"
done

echo "Dataset collection complete!"
echo "Designs: $(ls rtl_designs/*.v | wc -l)"
echo "Labels: $(ls labels/*.json | wc -l)"
```

## Using the Data with GraphRTL

Once you have collected the data:

```python
from pathlib import Path
from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml.feature_extraction import add_graph_features_to_data
import json
import torch

# Load your collected data
verilog_dir = Path("rtl_designs")
label_dir = Path("labels")

dataset = []

for verilog_file in verilog_dir.glob("*.v"):
    design_name = verilog_file.stem
    label_file = label_dir / f"{design_name}.json"
    
    if not label_file.exists():
        print(f"Warning: No label for {design_name}")
        continue
    
    # Convert to graph
    data = convert_verilog_to_pyg(str(verilog_file))
    data = add_graph_features_to_data(data)
    
    # Add label
    with open(label_file) as f:
        labels = json.load(f)
    data.y = torch.tensor([labels["Power"]], dtype=torch.float)
    
    dataset.append(data)
    print(f"Loaded {design_name}: {data.x.shape[0]} nodes, power={labels['Power']:.2f}")

print(f"\nTotal dataset size: {len(dataset)} designs")
```

## Troubleshooting

**Q: I don't have access to commercial EDA tools**
- Use open-source tools (Yosys, OpenSTA)
- Use published benchmarks with existing labels
- Start with simulation-based estimates
- Consider analytical models as approximation

**Q: Power values vary between synthesis runs**
- Ensure consistent synthesis settings
- Use same switching activity assumptions
- Run multiple times and average
- Document your methodology

**Q: How accurate should labels be?**
- Commercial EDA tools: ±5-10% accuracy
- Open-source estimates: ±20-30% accuracy
- Even approximate labels help the model learn patterns
- Can refine with better labels later

## Summary

**Recommended Pipeline:**
1. Collect 50-100 diverse RTL designs
2. Synthesize with consistent settings
3. Extract power from synthesis reports
4. Create JSON label files
5. Validate a few designs manually
6. Train your GNN model

**Time Estimate:**
- Setup EDA tools: 1-2 days
- Collect/synthesize 100 designs: 1-2 weeks
- Create automation scripts: 2-3 days
- Initial model training: 1 day

**Remember:** Start small! Even 20-30 designs are enough to validate your approach.
