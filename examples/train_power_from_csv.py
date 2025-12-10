#!/usr/bin/env python3
"""
Train GNN model for power estimation using CSV data with multiple clock periods per design.

This script handles the ISCAS benchmark data format where each design has multiple
runs at different clock periods, with power consumption measured for each run.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import click
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graphrtl.ml.feature_extraction import add_graph_features_to_data
from graphrtl.ml.gnn_model import HybridPowerGNN, PowerEstimationGNN
from graphrtl.ml.training import PowerEstimationTrainer
from graphrtl.sog.ast_to_pyg import ASTToPyG


def load_power_data_from_csv(csv_path: Path) -> dict:
    """
    Load power data from CSV file.

    Expected format:
        Design,Clock_Period,Total_Power,Leakage_Power
        iscas/s1196,200,4.62e-04,3.57e-08
        iscas/s1196,250,3.47e-04,3.26e-08
        ...

    Args:
        csv_path: Path to CSV file

    Returns:
        Dict mapping design_name -> list of (clock_period, total_power, leakage_power) tuples

    """
    power_data = defaultdict(list)

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            design = row["Design"]
            # Extract just the design name (e.g., 's1196' from 'iscas/s1196')
            design_name = design.split("/")[-1] if "/" in design else design

            clock_period = row["Clock_Period"]
            # Skip rows with unknown clock period
            if clock_period.lower() == "unknown":
                continue

            try:
                clock_period = float(clock_period)
                total_power = float(row["Total_Power"])
                leakage_power = float(row["Leakage_Power"])

                power_data[design_name].append(
                    (clock_period, total_power, leakage_power)
                )
            except (ValueError, KeyError) as e:
                click.echo(f"Warning: Skipping row for {design}: {e}")
                continue

    return power_data


def find_sog_file(design_name: str, sog_dir: Path) -> Path | None:
    """
    Find the SOG Verilog file for a design.

    Args:
        design_name: Name of the design (e.g., 's1196')
        sog_dir: Directory containing SOG files

    Returns:
        Path to SOG file, or None if not found

    """
    # Try exact match
    patterns = [
        f"{design_name}_bench.sog.v",
        f"{design_name}.sog.v",
        f"{design_name}_bench.sog.sv",
        f"{design_name}.sog.sv",
    ]

    for pattern in patterns:
        sog_file = sog_dir / pattern
        if sog_file.exists():
            return sog_file

    return None


def create_dataset_from_csv(
    csv_path: Path,
    sog_dir: Path,
    add_features: bool = True,
) -> list[Data]:
    """
    Create dataset from CSV file and SOG Verilog files.

    Each row in the CSV becomes one training sample with:
    - Node features from the SOG graph
    - Clock period as an additional input feature (normalized)
    - Total power as the target label

    Args:
        csv_path: Path to CSV file with power data
        sog_dir: Directory containing SOG Verilog files
        add_features: Whether to add handcrafted graph features

    Returns:
        List of PyG Data objects

    """
    dataset = []

    # Load power data from CSV
    click.echo(f"Loading power data from {csv_path}...")
    power_data = load_power_data_from_csv(csv_path)
    click.echo(f"Found power data for {len(power_data)} designs")

    # For each design, load SOG file and create samples
    for design_name, power_runs in power_data.items():
        sog_file = find_sog_file(design_name, sog_dir)

        if sog_file is None:
            click.echo(f"Warning: No SOG file found for {design_name}")
            continue

        click.echo(f"\nProcessing {design_name} ({len(power_runs)} runs)...")

        try:
            # Convert SOG Verilog to PyG graph (once per design)
            converter = ASTToPyG()
            data_template = converter.convert_verilog(sog_file)

            # Add handcrafted features if requested
            if add_features:
                data_template = add_graph_features_to_data(
                    data_template, converter.graph
                )

            click.echo(
                f"  Graph: {data_template.x.shape[0]} nodes, {data_template.edge_index.shape[1]} edges"
            )

            # Create one sample per power run (different clock periods)
            for clock_period, total_power, leakage_power in power_runs:
                # Clone the graph data
                data = data_template.clone()

                # Add clock period as a graph-level feature
                # Normalize clock period (picoseconds -> nanoseconds)
                clock_period_ns = clock_period / 1000.0
                data.clock_period = torch.tensor([clock_period_ns], dtype=torch.float)

                # Store leakage power as well (could be used for multi-task learning)
                data.leakage_power = torch.tensor([leakage_power], dtype=torch.float)

                # Set total power as the target
                data.y = torch.tensor([total_power], dtype=torch.float)

                # Store design name for reference
                data.design_name = design_name

                dataset.append(data)

            click.echo(f"  Created {len(power_runs)} training samples")

        except Exception as e:
            click.echo(f"Error processing {design_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    click.echo(f"\nTotal dataset size: {len(dataset)} samples")
    return dataset


def split_dataset_by_design(
    dataset: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Split dataset into train/val/test by design (not by individual samples).
    This ensures all runs of a design stay together.

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

    # Group samples by design
    design_groups = defaultdict(list)
    for data in dataset:
        design_groups[data.design_name].append(data)

    # Get list of unique designs
    designs = list(design_groups.keys())
    random.shuffle(designs)

    # Split designs
    n_designs = len(designs)
    train_size = int(n_designs * train_ratio)
    val_size = int(n_designs * val_ratio)

    train_designs = designs[:train_size]
    val_designs = designs[train_size : train_size + val_size]
    test_designs = designs[train_size + val_size :]

    # Collect all samples for each split
    train_data = []
    val_data = []
    test_data = []

    for design in train_designs:
        train_data.extend(design_groups[design])
    for design in val_designs:
        val_data.extend(design_groups[design])
    for design in test_designs:
        test_data.extend(design_groups[design])

    return train_data, val_data, test_data


class ClockAwareGNN(torch.nn.Module):
    """
    GNN model that incorporates clock period as an input feature.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        clock_mlp_hidden: int = 16,
    ):
        super().__init__()
        self.base_model = base_model

        # MLP to process clock period
        self.clock_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, clock_mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(clock_mlp_hidden, clock_mlp_hidden),
        )

        # Get output size from base model
        # The base model outputs to 1 dimension, we'll intercept before that
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(clock_mlp_hidden + 1, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
        )

    def forward(self, data):
        # Get graph embedding from base model (before final layer)
        # For this, we need to modify how we use the base model
        # Simpler approach: use base model as-is and combine with clock
        graph_pred = self.base_model(data)

        # Process clock period
        clock_embedding = self.clock_mlp(data.clock_period)

        # Pool clock embedding to batch level if needed
        if hasattr(data, "batch"):
            # Average pool across batch
            batch_size = data.batch.max().item() + 1
            clock_pooled = torch.zeros(
                batch_size, clock_embedding.shape[1], device=clock_embedding.device
            )
            for i in range(batch_size):
                clock_pooled[i] = clock_embedding[data.batch == i].mean(dim=0)
            clock_embedding = clock_pooled

        # Combine graph prediction with clock embedding
        combined = torch.cat([graph_pred, clock_embedding], dim=1)

        # Final prediction
        output = self.final_mlp(combined)
        return output


@click.command()
@click.option(
    "--csv-path",
    type=click.Path(exists=True, path_type=Path),
    default="power.csv",
    help="Path to CSV file with power data",
)
@click.option(
    "--sog-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".",
    help="Directory containing SOG Verilog files",
)
@click.option(
    "--model-type",
    type=click.Choice(["pure", "hybrid", "clock-aware"]),
    default="hybrid",
    help="Type of model to train",
)
@click.option("--epochs", type=int, default=100, help="Number of training epochs")
@click.option("--batch-size", type=int, default=16, help="Batch size for training")
@click.option(
    "--hidden-channels", type=int, default=64, help="Number of hidden channels"
)
@click.option("--num-layers", type=int, default=3, help="Number of GNN layers")
@click.option(
    "--conv-type",
    type=click.Choice(["gcn", "gat", "sage", "gin"]),
    default="gcn",
    help="Type of graph convolution",
)
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option(
    "--save-dir",
    type=click.Path(path_type=Path),
    default="models",
    help="Directory to save trained models",
)
def main(
    csv_path: Path,
    sog_dir: Path,
    model_type: str,
    epochs: int,
    batch_size: int,
    hidden_channels: int,
    num_layers: int,
    conv_type: str,
    lr: float,
    save_dir: Path,
):
    """Train GNN model for power estimation from CSV data."""
    click.echo("=" * 70)
    click.echo("GraphRTL GNN Power Estimation Training")
    click.echo("=" * 70)

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Create dataset
    add_features = model_type == "hybrid"
    dataset = create_dataset_from_csv(csv_path, sog_dir, add_features=add_features)

    if len(dataset) < 3:
        click.echo(
            f"Error: Not enough data (found {len(dataset)} samples). Need at least 3."
        )
        return

    # Split dataset by design (to avoid data leakage)
    train_data, val_data, test_data = split_dataset_by_design(dataset)

    click.echo("\n" + "=" * 70)
    click.echo("Dataset Split (by design):")
    click.echo("=" * 70)

    # Count unique designs in each split
    train_designs = set(d.design_name for d in train_data)
    val_designs = set(d.design_name for d in val_data)
    test_designs = set(d.design_name for d in test_data)

    click.echo(f"Train: {len(train_designs)} designs, {len(train_data)} samples")
    click.echo(f"  Designs: {sorted(train_designs)}")
    click.echo(f"Val:   {len(val_designs)} designs, {len(val_data)} samples")
    click.echo(f"  Designs: {sorted(val_designs)}")
    click.echo(f"Test:  {len(test_designs)} designs, {len(test_data)} samples")
    click.echo(f"  Designs: {sorted(test_designs)}")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Create model
    click.echo("\n" + "=" * 70)
    click.echo(f"Creating {model_type.upper()} model:")
    click.echo("=" * 70)

    if model_type == "pure":
        model = PowerEstimationGNN(
            in_channels=3,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=0.2,
            conv_type=conv_type,
            pooling="mean",
        )
    elif model_type == "hybrid":
        model = HybridPowerGNN(
            in_channels=3,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_graph_features=14,  # Handcrafted features
            dropout=0.2,
            conv_type=conv_type,
        )
    else:  # clock-aware
        base_model = HybridPowerGNN(
            in_channels=3,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_graph_features=14,
            dropout=0.2,
            conv_type=conv_type,
        )
        model = ClockAwareGNN(base_model, clock_mlp_hidden=16)

    click.echo(f"Model type: {model.__class__.__name__}")
    click.echo(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    click.echo(f"Conv type: {conv_type.upper()}")

    # Setup training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Device: {device}")

    trainer = PowerEstimationTrainer(model, device=device, lr=lr)

    # Create save directory
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f"power_gnn_{model_type}_{conv_type}.pt"

    # Train
    click.echo("\n" + "=" * 70)
    click.echo("Training:")
    click.echo("=" * 70)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path,
        early_stopping_patience=20,
    )

    # Evaluate on test set
    click.echo("\n" + "=" * 70)
    click.echo("Test Set Evaluation:")
    click.echo("=" * 70)

    metrics = trainer.evaluate(test_loader)
    click.echo(f"MSE:  {metrics['mse']:.6e}")
    click.echo(f"MAE:  {metrics['mae']:.6e}")
    click.echo(f"MAPE: {metrics['mape']:.2f}%")
    click.echo(f"R²:   {metrics.get('r2', 'N/A')}")

    # Show some example predictions
    click.echo("\n" + "=" * 70)
    click.echo("Example Predictions:")
    click.echo("=" * 70)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_data[:5]):  # Show first 5 test samples
            sample = sample.to(device)
            predicted = trainer.predict(sample)
            actual = sample.y.item()
            error_pct = abs(predicted - actual) / actual * 100

            clock_period = (
                sample.clock_period.item() if hasattr(sample, "clock_period") else "N/A"
            )

            click.echo(f"\nSample {i + 1}: {sample.design_name} @ {clock_period:.1f}ns")
            click.echo(f"  Predicted: {predicted:.6e} W")
            click.echo(f"  Actual:    {actual:.6e} W")
            click.echo(f"  Error:     {error_pct:.2f}%")

    click.echo("\n" + "=" * 70)
    click.echo(f"Model saved to: {save_path}")
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
