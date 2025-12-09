# Using PyG Graphs for GNN-Based Power Estimation

## Summary

I've set up a complete GNN-based power estimation framework for GraphRTL, inspired by MasterRTL's approach. Here's what has been created:

## What Was Created

### 1. **GNN Models** (`src/graphrtl/ml/gnn_model.py`)

Three model architectures:

- **PowerEstimationGNN**: Pure graph neural network
  - Learns from graph structure and node features
  - Supports GCN, GAT, GraphSAGE, and GIN convolutions
  - Configurable pooling (mean, max, add)

- **HybridPowerGNN** (Recommended): Combines GNN with handcrafted features
  - Best of both worlds: graph learning + domain knowledge
  - Uses 14 handcrafted features from MasterRTL's approach
  - Better performance with limited data

- **MultiTaskPowerGNN**: Predicts multiple PPA metrics
  - Simultaneous prediction of Power, Area, TNS, WNS
  - Shared representations across tasks
  - Multi-task learning benefits

### 2. **Feature Extraction** (`src/graphrtl/ml/feature_extraction.py`)

Utilities to extract MasterRTL-style features:

- `extract_graph_features()`: 14 graph-level features
  - seq_num, fanout_sum, io_num
  - Gate counts (AND, OR, NOT, XOR, MUX)
  - Area estimates (sequential, combinational)
  - Power estimates (static, dynamic)

- `calculate_analytical_power()`: Analytical power estimation
  - Similar to MasterRTL's approach
  - Useful for baseline comparison

- `add_graph_features_to_data()`: Adds features to PyG Data objects

### 3. **Training Utilities** (`src/graphrtl/ml/training.py`)

Complete training framework:

- **PowerEstimationTrainer**: Full training/validation/testing pipeline
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
  - Evaluation metrics (MSE, MAE, MAPE)

- `load_dataset_from_json()`: Load MasterRTL-format datasets

### 4. **Documentation** (`docs/gnn_power_estimation_guide.md`)

Comprehensive guide covering:
- Quick start examples
- Model selection guide
- Architecture comparisons
- Training tips
- Complete workflow example

### 5. **Example Script** (`examples/train_power_gnn.py`)

Practical training examples:
- Loading Verilog designs
- Creating datasets
- Training different models
- Comparing architectures

## How to Use It

### Basic Workflow

```python
from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml.gnn_model import HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.ml.feature_extraction import add_graph_features_to_data

# 1. Convert Verilog to graph
data = convert_verilog_to_pyg("design.v")
data = add_graph_features_to_data(data)
data.y = torch.tensor([power_label])  # Add label

# 2. Create model
model = HybridPowerGNN(
    in_channels=3,
    hidden_channels=64,
    num_layers=3,
    num_graph_features=14
)

# 3. Train
trainer = PowerEstimationTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)

# 4. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"MAPE: {metrics['mape']:.2f}%")

# 5. Predict
predicted_power = trainer.predict(new_data)
```

## Comparison: MasterRTL vs GraphRTL

| Aspect | MasterRTL | GraphRTL (GNN) |
|--------|-----------|----------------|
| **Model** | XGBoost | Graph Neural Network |
| **Input** | Graph-level features (14 values) | Node features + graph structure |
| **Pros** | Fast, interpretable, good baseline | Learns from topology, better generalization |
| **Cons** | Loses graph structure info | Needs more data, longer training |
| **Best For** | Quick estimation, limited data | Complex designs, large datasets |

### Hybrid Approach (Recommended)

The **HybridPowerGNN** combines both:
- GNN learns from graph structure
- Handcrafted features provide domain knowledge
- Best accuracy in practice

## Next Steps

### 1. **Collect Training Data**

You need:
- Verilog files for RTL designs
- Power labels (from synthesis tools)

Expected format:
```
designs/
  design1.v
  design2.v
labels/
  design1.json  # {"Power": 123.45, "Area": 678.9, ...}
  design2.json
```

### 2. **Train Your First Model**

```bash
# Set up dataset
mkdir -p example/verilog example/label

# Add your designs and labels

# Run training
python examples/train_power_gnn.py
```

### 3. **Tune Hyperparameters**

Key parameters to tune:
- `hidden_channels`: 32-128 (larger = more capacity)
- `num_layers`: 2-4 (deeper = more complex)
- `dropout`: 0.1-0.3 (higher = less overfitting)
- `conv_type`: Try 'gcn', 'gat', 'sage', 'gin'
- `lr`: 0.0001-0.01 (learning rate)

### 4. **Evaluate Performance**

Compare against:
- Analytical estimation (baseline)
- MasterRTL XGBoost model
- Actual synthesis results

### 5. **Improve the Model**

Advanced techniques:
- Add more node features (area per node, delay, etc.)
- Implement cross-validation
- Use ensemble methods
- Add data augmentation
- Implement attention mechanisms
- Use graph sampling for large designs

## Key Differences from MasterRTL

### What's Similar:
- ✓ Uses same feature extraction approach (14 features)
- ✓ Similar analytical power calculation
- ✓ Compatible with MasterRTL data format
- ✓ Same evaluation metrics

### What's New:
- ✓ Graph Neural Networks instead of XGBoost
- ✓ Learns from graph topology
- ✓ Node-level representations
- ✓ Multiple architecture options
- ✓ Multi-task learning support
- ✓ Hybrid approach (GNN + handcrafted features)

## Example: Training on TinyRocket

```python
from pathlib import Path
from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml.gnn_model import HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.ml.feature_extraction import add_graph_features_to_data
import torch

# Load TinyRocket design
verilog_dir = Path("MasterRTL/example/verilog")
data = convert_verilog_to_pyg(str(verilog_dir / "TinyRocket.v"))
data = add_graph_features_to_data(data)

# Add power label from MasterRTL
import json
with open("MasterRTL/example/label/TinyRocket.json") as f:
    labels = json.load(f)
data.y = torch.tensor([labels["Power"]], dtype=torch.float)

# Create and train model
model = HybridPowerGNN(in_channels=3, hidden_channels=64)
trainer = PowerEstimationTrainer(model)

# If you have multiple designs:
dataset = [...]  # List of Data objects
train_loader = DataLoader(dataset, batch_size=8)
trainer.train(train_loader, epochs=100)

# Predict power for new design
predicted_power = trainer.predict(data)
print(f"Predicted power: {predicted_power:.4f}")
```

## Tips for Success

1. **Start Simple**: Begin with PowerEstimationGNN and small datasets
2. **Use Validation**: Always split data into train/val/test
3. **Normalize Labels**: Scale power values to [0, 1] range
4. **Check Overfitting**: Monitor train vs validation loss
5. **Try Hybrid Model**: Usually performs best in practice
6. **Compare Baselines**: Compare against analytical estimation
7. **Visualize Results**: Plot predictions vs actual values
8. **Use Early Stopping**: Prevent overfitting with patience parameter

## Troubleshooting

**Q: Model not converging**
- Normalize input features and labels
- Try smaller learning rate (0.0001)
- Check for NaN/Inf in data
- Start with simpler model

**Q: Overfitting**
- Increase dropout (0.3-0.5)
- Add weight decay (1e-4)
- Use early stopping
- Get more training data

**Q: Poor accuracy**
- Try hybrid model (adds handcrafted features)
- Increase model capacity (more layers/channels)
- Check data quality
- Ensure labels are correct

## Resources

- **Documentation**: `docs/gnn_power_estimation_guide.md`
- **Examples**: `examples/train_power_gnn.py`
- **Models**: `src/graphrtl/ml/gnn_model.py`
- **MasterRTL Reference**: `../MasterRTL/`

## Getting Help

If you encounter issues:
1. Check the documentation
2. Review the example script
3. Compare with MasterRTL's approach
4. Verify your data format
5. Start with small experiments

Good luck with your power estimation model! 🚀
