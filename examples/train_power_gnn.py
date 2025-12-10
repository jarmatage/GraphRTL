#!/usr/bin/env python3
"""
Example script for training a GNN model for power estimation.

This demonstrates the complete workflow from converting Verilog designs
to training a GNN model for power consumption estimation.
"""

import json
import random
import traceback
from pathlib import Path

import click
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graphrtl.ml.feature_extraction import add_graph_features_to_data
from graphrtl.ml.gnn_model import HybridPowerGNN, PowerEstimationGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.sog.ast_to_pyg import ASTToPyG


def load_design_with_power_label(
    verilog_path: Path,
    label_file: Path,
    label_type: str = "Power",
) -> Data | None:
    """
    Load a Verilog design and its power label.

    Args:
        verilog_path: Path to Verilog file
        label_file: Path to JSON file with power labels
        label_type: Type of label ('Power', 'Area', 'TNS', 'WNS')

    Returns:
        PyG Data object with features and label, or None if loading fails

    """
    # Convert Verilog to PyG graph
    click.echo(f"Converting {verilog_path.name}...")
    converter = ASTToPyG()
    data = converter.convert_verilog(verilog_path)

    # Add handcrafted features (for hybrid model)
    data = add_graph_features_to_data(data, converter.graph)

    # Load power label
    with label_file.open() as f:
        labels = json.load(f)

    if label_type not in labels:
        click.echo(f"Warning: {label_type} not found in {label_file}")
        return None

    power = labels[label_type]
    data.y = torch.tensor([power], dtype=torch.float)

    click.echo(
        f"  Loaded: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges, power={power:.4f}"
    )
    return data


def create_dataset_from_directory(
    verilog_dir: Path,
    label_dir: Path,
    label_type: str = "Power",
) -> list[Data]:
    """
    Create dataset from directory of Verilog files and labels.

    Expected structure:
        verilog_dir/
            design1.v
            design2.v
            ...
        label_dir/
            design1.json  # {"Power": 123.45, "Area": 678.90, ...}
            design2.json
            ...

    Args:
        verilog_dir: Directory containing Verilog files
        label_dir: Directory containing label JSON files
        label_type: Type of label to load

    Returns:
        List of PyG Data objects

    """
    dataset = []

    verilog_files = list(verilog_dir.glob("*.v")) + list(verilog_dir.glob("*.sv"))

    for verilog_file in verilog_files:
        design_name = verilog_file.stem
        label_file = label_dir / f"{design_name}.json"

        if not label_file.exists():
            click.echo(f"Warning: No label file found for {design_name}")
            continue

        data = load_design_with_power_label(verilog_file, label_file, label_type)
        if data is not None:
            dataset.append(data)

    return dataset


def split_dataset(
    dataset: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: List of Data objects
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set

    Returns:
        Tuple of (train_data, val_data, test_data)

    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    random.shuffle(dataset)

    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]

    return train_data, val_data, test_data


def train_pure_gnn_example() -> None:
    """Train a pure GNN model."""
    click.echo("\n" + "=" * 70)
    click.echo("Training Pure GNN Model")
    click.echo("=" * 70)

    # Load dataset (replace with your actual paths)
    verilog_dir = Path("example/verilog")
    label_dir = Path("example/label")

    if not verilog_dir.exists() or not label_dir.exists():
        click.echo("Example directories not found. Skipping...")
        click.echo(
            "Create directories with Verilog files and labels to run this example."
        )
        return

    # Create dataset
    dataset = create_dataset_from_directory(verilog_dir, label_dir, label_type="Power")

    if len(dataset) < 3:
        click.echo(f"Not enough data (found {len(dataset)} designs). Need at least 3.")
        return

    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset)
    click.echo("\nDataset split:")
    click.echo(f"  Train: {len(train_data)} designs")
    click.echo(f"  Val:   {len(val_data)} designs")
    click.echo(f"  Test:  {len(test_data)} designs")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    test_loader = DataLoader(test_data, batch_size=8)

    # Create model
    model = PowerEstimationGNN(
        in_channels=3,
        hidden_channels=32,
        num_layers=3,
        dropout=0.2,
        conv_type="gcn",
        pooling="mean",
    )

    click.echo(f"\nModel: {model.__class__.__name__}")
    click.echo(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Device: {device}")

    trainer = PowerEstimationTrainer(model, device=device, lr=0.001)

    # Train
    save_path = Path("models/power_gnn_pure.pt")
    save_path.parent.mkdir(exist_ok=True)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        save_path=save_path,
        early_stopping_patience=10,
    )

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    click.echo("\nTest Results:")
    click.echo(f"  MSE:  {metrics['mse']:.6f}")
    click.echo(f"  MAE:  {metrics['mae']:.6f}")
    click.echo(f"  MAPE: {metrics['mape']:.2f}%")


def train_hybrid_gnn_example() -> None:
    """Train a hybrid GNN model (recommended)."""
    click.echo("\n" + "=" * 70)
    click.echo("Training Hybrid GNN Model (Recommended)")
    click.echo("=" * 70)

    # Load dataset
    verilog_dir = Path("example/verilog")
    label_dir = Path("example/label")

    if not verilog_dir.exists() or not label_dir.exists():
        click.echo("Example directories not found. Skipping...")
        return

    dataset = create_dataset_from_directory(verilog_dir, label_dir, label_type="Power")

    if len(dataset) < 3:
        click.echo(f"Not enough data (found {len(dataset)} designs).")
        return

    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset)
    click.echo("\nDataset split:")
    click.echo(f"  Train: {len(train_data)} designs")
    click.echo(f"  Val:   {len(val_data)} designs")
    click.echo(f"  Test:  {len(test_data)} designs")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    test_loader = DataLoader(test_data, batch_size=8)

    # Create hybrid model
    model = HybridPowerGNN(
        in_channels=3,
        hidden_channels=64,
        num_layers=3,
        num_graph_features=14,  # Handcrafted features
        dropout=0.2,
        conv_type="gcn",
    )

    click.echo(f"\nModel: {model.__class__.__name__}")
    click.echo(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Device: {device}")

    trainer = PowerEstimationTrainer(model, device=device, lr=0.001)

    # Train
    save_path = Path("models/power_gnn_hybrid.pt")
    save_path.parent.mkdir(exist_ok=True)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        save_path=save_path,
        early_stopping_patience=20,
    )

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    click.echo("\nTest Results:")
    click.echo(f"  MSE:  {metrics['mse']:.6f}")
    click.echo(f"  MAE:  {metrics['mae']:.6f}")
    click.echo(f"  MAPE: {metrics['mape']:.2f}%")

    # Make prediction on first test sample
    if test_data:
        sample = test_data[0]
        predicted = trainer.predict(sample)
        actual = sample.y.item()
        click.echo("\nExample Prediction:")
        click.echo(f"  Predicted: {predicted:.4f}")
        click.echo(f"  Actual:    {actual:.4f}")
        click.echo(
            f"  Error:     {abs(predicted - actual):.4f} ({abs(predicted - actual) / actual * 100:.2f}%)"
        )


def compare_architectures() -> None:
    """Compare different GNN architectures."""
    click.echo("\n" + "=" * 70)
    click.echo("Comparing GNN Architectures")
    click.echo("=" * 70)

    # Load dataset
    verilog_dir = Path("example/verilog")
    label_dir = Path("example/label")

    if not verilog_dir.exists() or not label_dir.exists():
        click.echo("Example directories not found. Skipping...")
        return

    dataset = create_dataset_from_directory(verilog_dir, label_dir)

    if len(dataset) < 10:
        click.echo(f"Not enough data for comparison (found {len(dataset)} designs).")
        return

    train_data, val_data, test_data = split_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    architectures = ["gcn", "gat", "sage", "gin"]
    results = {}

    for arch in architectures:
        click.echo(f"\nTraining with {arch.upper()}...")

        model = PowerEstimationGNN(
            in_channels=3, hidden_channels=32, num_layers=3, conv_type=arch
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = PowerEstimationTrainer(model, device=device, lr=0.001)

        # Quick training
        trainer.train(train_loader=train_loader, epochs=20)

        # Evaluate
        metrics = trainer.evaluate(test_loader)
        results[arch] = metrics

        click.echo(f"  MAE: {metrics['mae']:.6f}, MAPE: {metrics['mape']:.2f}%")

    # click.echo comparison
    click.echo("\n" + "=" * 70)
    click.echo("Architecture Comparison Results:")
    click.echo("=" * 70)
    click.echo(f"{'Architecture':<15} {'MAE':<12} {'MAPE':<10}")
    click.echo("-" * 40)
    for arch, metrics in sorted(results.items(), key=lambda x: x[1]["mae"]):
        click.echo(
            f"{arch.upper():<15} {metrics['mae']:<12.6f} {metrics['mape']:<10.2f}%"
        )


def main() -> None:
    """Run examples."""
    click.echo("=" * 70)
    click.echo("GraphRTL GNN Power Estimation Examples")
    click.echo("=" * 70)
    click.echo("\nBased on MasterRTL's approach, adapted for GNN models.")
    click.echo("\nIMPORTANT: Set up your dataset first:")
    click.echo("  1. Place Verilog files in example/verilog/")
    click.echo("  2. Place label JSON files in example/label/")
    click.echo('     Format: {"Power": 123.45, "Area": 678.90, ...}')

    # Run examples
    try:
        # Example 1: Pure GNN
        train_pure_gnn_example()

        # Example 2: Hybrid GNN (recommended)
        train_hybrid_gnn_example()

        # Example 3: Compare architectures
        # compare_architectures()

    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted by user.")
    except Exception as e:
        click.echo(f"\n\nError during training: {e}")

        traceback.click.echo_exc()

    click.echo("\n" + "=" * 70)
    click.echo("Examples complete!")
    click.echo("=" * 70)
    click.echo("\nNext steps:")
    click.echo("  1. Collect more training data")
    click.echo("  2. Tune hyperparameters")
    click.echo("  3. Try different architectures")
    click.echo("  4. Implement cross-validation")
    click.echo("\nSee docs/gnn_power_estimation_guide.md for more details.")


if __name__ == "__main__":
    main()
