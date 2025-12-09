# GNN-based Power Estimation for RTL Designs

This guide explains how to use GraphRTL's GNN models for power estimation of RTL designs, based on the MasterRTL approach.

## Overview

GraphRTL provides three types of GNN models for power estimation:

1. **PowerEstimationGNN**: Pure GNN that learns from graph structure and node features
2. **HybridPowerGNN**: Combines GNN with handcrafted features (recommended)
3. **MultiTaskPowerGNN**: Learns multiple PPA metrics simultaneously

## Quick Start

### 1. Convert Verilog to Graph

```python
from graphrtl.sog import convert_verilog_to_pyg

# Convert your RTL design to a PyG graph
data = convert_verilog_to_pyg("path/to/design.v")

# The graph now has:
# - data.x: Node features [bit_width, fanout, toggle_rate]
# - data.edge_index: Edge connections
# - data.node_names: Names of nodes
# - data.node_types: Types of nodes
```

### 2. Extract Features (Optional for Hybrid Model)

```python
from graphrtl.ml.feature_extraction import add_graph_features_to_data
from graphrtl.sog import ASTToPyG

# Add handcrafted graph-level features
converter = ASTToPyG()
data_with_features = add_graph_features_to_data(data, converter.graph)

# Now data.graph_features contains 14 features:
# [seq_num, fanout_sum, io_num, and_num, or_num, not_num, xor_num, mux_num,
#  seq_area, comb_area, total_area, stat_pwr, dyn_pwr, total_pwr]
```

### 3. Create and Train Model

```python
from graphrtl.ml.gnn_model import PowerEstimationGNN, HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer
from torch_geometric.loader import DataLoader

# Option 1: Pure GNN
model = PowerEstimationGNN(
    in_channels=3,           # bit_width, fanout, toggle_rate
    hidden_channels=64,
    num_layers=3,
    dropout=0.2,
    conv_type='gcn',         # 'gcn', 'gat', 'sage', or 'gin'
    pooling='mean'           # 'mean', 'max', or 'add'
)

# Option 2: Hybrid GNN (recommended)
model = HybridPowerGNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=3,
    num_graph_features=14,   # Number of handcrafted features
    dropout=0.2,
    conv_type='gcn'
)

# Create trainer
trainer = PowerEstimationTrainer(
    model=model,
    device='cuda',           # or 'cpu'
    lr=0.001,
    weight_decay=1e-5
)

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train the model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path='models/power_gnn.pt',
    early_stopping_patience=20
)
```

### 4. Evaluate and Predict

```python
# Evaluate on test set
test_loader = DataLoader(test_dataset, batch_size=32)
metrics = trainer.evaluate(test_loader)

print(f"Test MSE: {metrics['mse']:.6f}")
print(f"Test MAE: {metrics['mae']:.6f}")
print(f"Test MAPE: {metrics['mape']:.2f}%")

# Make prediction for a single design
predicted_power = trainer.predict(data)
print(f"Predicted power: {predicted_power:.6f}")
```

## Comparison with MasterRTL Approach

### MasterRTL (Traditional ML)
- Uses **XGBoost** on graph-level features
- Features: operator counts, area estimates, toggle rates
- Pros: Fast training, interpretable, good baseline
- Cons: Loses graph structure information

### GraphRTL (GNN)
- Uses **Graph Neural Networks** on node-level representation
- Features: Node features + graph structure
- Pros: Learns from graph topology, better generalization
- Cons: Requires more training data, longer training time

### Hybrid Approach (Recommended)
- Combines both approaches
- Uses GNN + handcrafted features
- Best of both worlds

## Feature Engineering

GraphRTL provides utilities to extract features similar to MasterRTL:

```python
from graphrtl.ml.feature_extraction import (
    extract_graph_features,
    calculate_analytical_power
)

# Extract 14 graph-level features (like MasterRTL's cal_oper)
features = extract_graph_features(directed_graph)
# Returns: [seq_num, fanout_sum, io_num, and_num, or_num, not_num, 
#           xor_num, mux_num, seq_area, comb_area, total_area, 
#           stat_pwr, dyn_pwr, total_pwr]

# Calculate analytical power estimate
power_est = calculate_analytical_power(directed_graph, toggle_rates)
# Returns: {'static_power': x, 'dynamic_power': y, 'total_power': z}
```

## Advanced: Multi-Task Learning

Train a model that predicts multiple PPA metrics simultaneously:

```python
from graphrtl.ml.gnn_model import MultiTaskPowerGNN

model = MultiTaskPowerGNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=3,
    dropout=0.2
)

# Forward pass returns multiple predictions
predictions = model(data)
# Returns: {
#     'power': predicted_power,
#     'area': predicted_area,
#     'tns': predicted_tns,
#     'wns': predicted_wns
# }
```

## Model Selection Guide

### Use **PowerEstimationGNN** if:
- You have large datasets (>1000 designs)
- You want to learn purely from graph structure
- You have diverse designs with varying topologies

### Use **HybridPowerGNN** if:
- You have medium datasets (100-1000 designs)
- You want best accuracy (recommended)
- You want to combine traditional and deep learning

### Use **MultiTaskPowerGNN** if:
- You have labels for multiple metrics (power, area, timing)
- You want to leverage shared representations
- You care about multiple PPA metrics

## GNN Architecture Choices

### Convolution Types
- **GCN** (Graph Convolutional Network): Standard choice, fast
- **GAT** (Graph Attention Network): Learns edge importance, more expressive
- **GraphSAGE**: Handles large graphs well, good for varying sizes
- **GIN** (Graph Isomorphism Network): Most expressive, slower

### Pooling Methods
- **mean**: Average over all nodes (default, stable)
- **max**: Maximum over all nodes (captures extremes)
- **add**: Sum over all nodes (depends on graph size)

## Training Tips

1. **Start Small**: Begin with a small model (hidden_channels=32, num_layers=2)
2. **Use Validation**: Always use a validation set to prevent overfitting
3. **Learning Rate**: Start with 0.001, reduce if loss plateaus
4. **Batch Size**: Larger is better for stability, but use what fits in memory
5. **Data Augmentation**: If data is limited, consider graph augmentation
6. **Normalize Labels**: Normalize power values to [0, 1] range for better training

## Example Workflow

```python
import torch
from pathlib import Path
from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml.gnn_model import HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.ml.feature_extraction import add_graph_features_to_data
from torch_geometric.loader import DataLoader

# 1. Load and convert designs
design_files = list(Path("designs/").glob("*.v"))
dataset = []

for verilog_file in design_files:
    # Convert to graph
    data = convert_verilog_to_pyg(str(verilog_file))
    
    # Add handcrafted features
    data = add_graph_features_to_data(data)
    
    # Add label (you need to load this from your data)
    data.y = torch.tensor([power_label], dtype=torch.float)
    
    dataset.append(data)

# 2. Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]

# 3. Create data loaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# 4. Create and train model
model = HybridPowerGNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=3,
    num_graph_features=14,
    dropout=0.2
)

trainer = PowerEstimationTrainer(model, device='cuda', lr=0.001)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path=Path("models/power_gnn_best.pt"),
    early_stopping_patience=20
)

# 5. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Test Results:")
print(f"  MSE:  {metrics['mse']:.6f}")
print(f"  MAE:  {metrics['mae']:.6f}")
print(f"  MAPE: {metrics['mape']:.2f}%")

# 6. Make predictions on new designs
new_data = convert_verilog_to_pyg("new_design.v")
new_data = add_graph_features_to_data(new_data)
predicted_power = trainer.predict(new_data)
print(f"Predicted power for new design: {predicted_power:.6f}")
```

## Next Steps

1. **Collect training data** (see [Data Collection Guide](collecting_training_data.md))
   - Gather RTL designs from open-source projects or internal IP
   - Synthesize designs and extract power labels using EDA tools
   - Aim for 50-100+ designs with diverse characteristics
2. Experiment with different model architectures
3. Add more node features (area per node, delay, etc.)
4. Implement cross-validation for better evaluation
5. Add data augmentation techniques
6. Implement ensemble methods (combine multiple models)

## References

- MasterRTL: Traditional ML approach with handcrafted features
- PyTorch Geometric: Framework for GNNs
- GraphRTL: This library

## Troubleshooting

**Q: Model not learning (loss not decreasing)**
- Check if labels are normalized
- Try smaller learning rate
- Check for nan/inf in data
- Try simpler model first

**Q: Model overfitting (val loss increasing)**
- Increase dropout
- Add weight decay
- Use early stopping
- Get more training data

**Q: Out of memory errors**
- Reduce batch size
- Reduce hidden_channels
- Reduce num_layers
- Use CPU if GPU memory is insufficient
