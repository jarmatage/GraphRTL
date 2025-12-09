# Quick Reference: Data Collection for Power Estimation

## TL;DR - Three Main Options

### Option 1: Commercial EDA Tools (Best Accuracy)
```bash
# Synopsys Design Compiler or Cadence Genus
# Synthesize → Report Power → Extract to JSON
✓ Most accurate (±5-10%)
✗ Requires expensive licenses
```

### Option 2: Open-Source Tools (Free)
```bash
# Yosys + OpenSTA + Python estimation
yosys -p "synth; stat" design.v
✓ Free and open-source
✗ Less accurate (±20-30%)
```

### Option 3: Use Existing Datasets (Fastest Start)
```bash
# Use MasterRTL examples or published benchmarks
✓ Ready to use immediately
✗ Limited to available designs
```

## Minimal Working Example

```python
#!/usr/bin/env python3
"""Create a minimal dataset from open-source designs."""

import json
import subprocess
from pathlib import Path

# 1. Get some open-source RTL
designs = [
    "https://raw.githubusercontent.com/YosysHQ/picorv32/master/picorv32.v",
    # Add more URLs
]

for url in designs:
    subprocess.run(["wget", url, "-P", "rtl_designs/"])

# 2. For each design, create a label (simplified)
for verilog_file in Path("rtl_designs").glob("*.v"):
    # Simple estimation based on lines of code
    num_lines = len(verilog_file.read_text().splitlines())
    estimated_power = num_lines * 0.1  # Very rough estimate
    
    label = {
        "Power": estimated_power,
        "Area": num_lines * 10,
        "source": "estimation"
    }
    
    label_file = Path("labels") / f"{verilog_file.stem}.json"
    label_file.parent.mkdir(exist_ok=True)
    with open(label_file, "w") as f:
        json.dump(label, f)

print("Done! Start with this and refine with actual synthesis later.")
```

## Data Structure

```
your_project/
├── rtl_designs/          # Your Verilog files
│   ├── design1.v
│   ├── design2.v
│   └── design3.sv
└── labels/               # Power labels (JSON)
    ├── design1.json     # {"Power": 123.45, "Area": 678.9}
    ├── design2.json
    └── design3.json
```

## Getting Real Power Labels

### With Synopsys DC:
```tcl
# In dc_shell:
compile_ultra
report_power > power.txt
# Parse power.txt: "Total Power = XXX mW"
```

### With Yosys (Free):
```bash
yosys -p "
    read_verilog design.v;
    synth;
    stat
" | grep -E "Number of cells|wires"
# Estimate: cells * 0.1mW + wires * 0.01mW
```

### From Simulation:
```bash
iverilog design.v testbench.v
vvp a.out
# Analyze VCD for switching activity
python estimate_power_from_vcd.py dump.vcd
```

## Quick Validation

Check your dataset is ready:
```python
from pathlib import Path
import json

verilog_count = len(list(Path("rtl_designs").glob("*.v")))
label_count = len(list(Path("labels").glob("*.json")))

print(f"RTL files: {verilog_count}")
print(f"Labels: {label_count}")

if verilog_count != label_count:
    print("⚠️  Mismatch! Some designs missing labels")
else:
    print("✓ Dataset ready!")
    
# Check first label
first_label = list(Path("labels").glob("*.json"))[0]
with open(first_label) as f:
    data = json.load(f)
    print(f"Sample label: {data}")
```

## Recommended Starting Point

1. **Start with 20-30 designs** from open-source projects
2. **Use Yosys** for free synthesis and rough estimates
3. **Train initial model** to validate approach
4. **Refine with better tools** as needed
5. **Gradually expand** to 100+ designs

## Open-Source Design Sources

- **GitHub**: Search "verilog", "risc-v", "processor"
- **OpenCores**: opencores.org
- **ChipsAlliance**: github.com/chipsalliance
- **LowRISC**: github.com/lowrisc
- **Benchmarks**: ISCAS, IWLS, EPFL

## Time Investment

- **Quick start (estimates)**: 1 day → 20 designs
- **With synthesis (Yosys)**: 1 week → 50 designs  
- **With EDA tools**: 2 weeks → 100 designs
- **Large dataset**: 1-2 months → 500+ designs

## Need Help?

See the full guide: [docs/collecting_training_data.md](collecting_training_data.md)
