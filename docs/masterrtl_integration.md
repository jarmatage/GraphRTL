# Integration with MasterRTL Data

This guide shows how to use your existing MasterRTL data with GraphRTL's GNN models.

## Using MasterRTL Features

If you already have MasterRTL features extracted, you can use them directly:

```python
import json
import torch
from torch_geometric.data import Data

# Load MasterRTL features
with open("MasterRTL/example/feature/TinyRocket_sog_vec_area.json") as f:
    features = json.load(f)

# Load label
with open("MasterRTL/example/label/TinyRocket.json") as f:
    labels = json.load(f)

# Create simple Data object with features
# (You'll need to also load the actual graph structure separately)
graph_features = torch.tensor(features, dtype=torch.float)

# For hybrid model, attach to your graph data
data.graph_features = graph_features
data.y = torch.tensor([labels["Power"]], dtype=torch.float)
```

## Using MasterRTL SOG Graphs

If you have MasterRTL's pickle files with graph structure:

```python
import pickle
import networkx as nx
from graphrtl.ml.feature_extraction import extract_graph_features
from graphrtl.sog import DirectedGraph

# Load MasterRTL graph
with open("MasterRTL/example/sog/TinyRocket_sog.pkl", "rb") as f:
    graph_dict = pickle.load(f)

with open("MasterRTL/example/sog/TinyRocket_sog_node_dict.pkl", "rb") as f:
    node_dict = pickle.load(f)

# Convert to GraphRTL DirectedGraph
graph = DirectedGraph()
graph.graph = graph_dict
graph.node_dict = node_dict

# Extract features using GraphRTL
features = extract_graph_features(graph)
print("Extracted features:", features)
```

## Batch Processing MasterRTL Designs

Process multiple designs from MasterRTL format:

```python
from pathlib import Path
from graphrtl.ml.training import load_dataset_from_json

# Load all designs from MasterRTL directories
design_names = [
    "TinyRocket",
    # Add more design names
]

feat_dir = Path("MasterRTL/example/feature")
label_dir = Path("MasterRTL/example/label")

# Load dataset
dataset = load_dataset_from_json(
    design_names=design_names,
    feat_dir=feat_dir,
    label_dir=label_dir,
    label_type="Power"
)

print(f"Loaded {len(dataset)} designs")
```

## Converting MasterRTL Pipeline to GraphRTL

Here's how MasterRTL steps map to GraphRTL:

### MasterRTL Pipeline:
1. Verilog → Yosys SOG → Pickle
2. Feature extraction (area, power, timing)
3. XGBoost training
4. Inference

### GraphRTL GNN Pipeline:
1. Verilog → PyG Graph (via ast_to_pyg)
2. Feature extraction (same 14 features + node features)
3. GNN training
4. Inference

## Example: Full Migration

```python
#!/usr/bin/env python3
"""Migrate from MasterRTL to GraphRTL GNN."""

import json
import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data, DataLoader

from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml import HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.ml.feature_extraction import add_graph_features_to_data

def load_masterrtl_design(design_name, verilog_dir, label_dir):
    """Load a design from MasterRTL format."""
    # Convert Verilog to PyG
    verilog_path = verilog_dir / f"{design_name}.v"
    data = convert_verilog_to_pyg(str(verilog_path))
    
    # Add graph features (14 features like MasterRTL)
    data = add_graph_features_to_data(data)
    
    # Load power label
    label_path = label_dir / f"{design_name}.json"
    with open(label_path) as f:
        labels = json.load(f)
    
    data.y = torch.tensor([labels["Power"]], dtype=torch.float)
    return data

def main():
    # Paths
    verilog_dir = Path("MasterRTL/example/verilog")
    label_dir = Path("MasterRTL/example/label")
    
    # Design names
    designs = [
        "TinyRocket",
        # Add more designs here
    ]
    
    # Load all designs
    dataset = []
    for design in designs:
        try:
            data = load_masterrtl_design(design, verilog_dir, label_dir)
            dataset.append(data)
            print(f"Loaded {design}: {data.x.shape[0]} nodes")
        except Exception as e:
            print(f"Error loading {design}: {e}")
    
    if len(dataset) < 3:
        print("Need at least 3 designs for train/val/test split")
        return
    
    # Split dataset
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    print(f"\nDataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4)
    test_loader = DataLoader(test_data, batch_size=4)
    
    # Create hybrid model (uses both GNN and handcrafted features)
    model = HybridPowerGNN(
        in_channels=3,          # bit_width, fanout, toggle_rate
        hidden_channels=64,
        num_layers=3,
        num_graph_features=14,  # MasterRTL-style features
        dropout=0.2
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train
    trainer = PowerEstimationTrainer(model, device='cuda', lr=0.001)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        save_path=Path("models/masterrtl_migration.pt"),
        early_stopping_patience=20
    )
    
    # Evaluate
    metrics = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Compare with MasterRTL XGBoost results
    print("\nTo compare with MasterRTL:")
    print("  1. Run MasterRTL's XGBoost on same designs")
    print("  2. Compare MAPE metrics")
    print("  3. GNN typically performs better with more data")

if __name__ == "__main__":
    main()
```

## Feature Comparison

| Feature | MasterRTL | GraphRTL GNN |
|---------|-----------|--------------|
| seq_num | ✓ | ✓ |
| fanout_sum | ✓ | ✓ |
| io_num | ✓ | ✓ |
| and_num | ✓ | ✓ |
| or_num | ✓ | ✓ |
| not_num | ✓ | ✓ |
| xor_num | ✓ | ✓ |
| mux_num | ✓ | ✓ |
| seq_area | ✓ | ✓ |
| comb_area | ✓ | ✓ |
| total_area | ✓ | ✓ |
| stat_pwr | ✓ | ✓ |
| dyn_pwr | ✓ | ✓ |
| total_pwr | ✓ | ✓ |
| **Graph structure** | ✗ | ✓ |
| **Node embeddings** | ✗ | ✓ |

## Performance Expectations

Based on typical results:

**MasterRTL (XGBoost)**:
- MAPE: 5-15% (with good features)
- Training time: Minutes
- Inference: Milliseconds

**GraphRTL (GNN)**:
- MAPE: 3-12% (hybrid model)
- Training time: Hours (depends on dataset size)
- Inference: Seconds

**GraphRTL Hybrid** typically performs best when:
- You have 50+ training designs
- Designs have diverse topologies
- Graph structure matters (hierarchical designs)

## Tips for Migration

1. **Start with Hybrid Model**: Uses both GNN and MasterRTL features
2. **Use Same Train/Test Split**: For fair comparison
3. **Normalize Labels**: Both models benefit from normalized power values
4. **Compare Metrics**: Use same evaluation metrics (MAPE, MAE)
5. **Ensemble Both**: Can combine GNN and XGBoost predictions

## When to Use Which

**Use MasterRTL XGBoost if**:
- You have limited data (<50 designs)
- You need fast training
- You need interpretable features
- You want a quick baseline

**Use GraphRTL GNN if**:
- You have more data (>100 designs)
- Graph structure is important
- You want better accuracy
- You can afford longer training

**Use Hybrid (Recommended)**:
- Best of both worlds
- Works well with medium datasets (50-500 designs)
- Combines GNN learning with domain knowledge
