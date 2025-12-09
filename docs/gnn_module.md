# GraphRTL Machine Learning Module

This module provides GNN-based models for RTL power estimation, inspired by MasterRTL's approach.

## Quick Start

```python
from graphrtl.sog import convert_verilog_to_pyg
from graphrtl.ml import HybridPowerGNN
from graphrtl.ml.training import PowerEstimationTrainer

# Convert design to graph
data = convert_verilog_to_pyg("design.v")

# Train model
model = HybridPowerGNN()
trainer = PowerEstimationTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)

# Predict
power = trainer.predict(data)
```

## Module Contents

- **gnn_model.py**: GNN architectures (PowerEstimationGNN, HybridPowerGNN, MultiTaskPowerGNN)
- **training.py**: Training utilities and PowerEstimationTrainer
- **feature_extraction.py**: Feature extraction inspired by MasterRTL

## Documentation

See `docs/gnn_power_estimation_guide.md` for detailed guide.

## Example

See `examples/train_power_gnn.py` for complete training example.
