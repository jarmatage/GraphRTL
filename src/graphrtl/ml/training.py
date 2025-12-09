"""Training utilities for GNN models."""

import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .gnn_model import HybridPowerGNN, PowerEstimationGNN

logger = logging.getLogger(__name__)


class PowerEstimationTrainer:
    """
    Trainer for power estimation GNN models.

    Example usage:
        trainer = PowerEstimationTrainer(model, device='cuda')
        trainer.train(train_loader, val_loader, epochs=100)
        metrics = trainer.evaluate(test_loader)

    """

    def __init__(
        self,
        model: PowerEstimationGNN | HybridPowerGNN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 0.001,
        weight_decay: float = 1e-5,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: GNN model to train
            device: Device to train on ('cuda' or 'cpu')
            lr: Learning rate
            weight_decay: Weight decay for optimizer

        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss for the epoch

        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for starting_batch in train_loader:
            batch = starting_batch.to(self.device)
            self.optimizer.zero_grad()

            # Check if we have labels
            if not hasattr(batch, "y") or batch.y is None:
                continue

            # Forward pass
            if isinstance(self.model, HybridPowerGNN):
                # Hybrid model needs graph features
                if hasattr(batch, "graph_features"):
                    pred = self.model(batch, batch.graph_features)
                else:
                    continue
            else:
                pred = self.model(batch)

            # Calculate loss
            loss = self.criterion(pred, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss

        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for starting_batch in val_loader:
                batch = starting_batch.to(self.device)

                if not hasattr(batch, "y") or batch.y is None:
                    continue

                # Forward pass
                if isinstance(self.model, HybridPowerGNN):
                    if hasattr(batch, "graph_features"):
                        pred = self.model(batch, batch.graph_features)
                    else:
                        continue
                else:
                    pred = self.model(batch)

                # Calculate loss
                loss = self.criterion(pred, batch.y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        save_path: Path | None = None,
        early_stopping_patience: int = 20,
    ) -> dict[str, list[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            save_path: Path to save the best model
            early_stopping_patience: Number of epochs to wait for improvement

        Returns:
            Dictionary with training history

        """
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

                logger.info(
                    "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                        logger.info("Saved best model to %s", save_path)
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping after %d epochs", epoch + 1)
                    break
            else:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.6f", epoch + 1, epochs, train_loss
                )

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with evaluation metrics

        """
        self.model.eval()
        predictions = []
        targets = []
        losses = []

        with torch.no_grad():
            for starting_batch in test_loader:
                batch = starting_batch.to(self.device)

                if not hasattr(batch, "y") or batch.y is None:
                    continue

                # Forward pass
                if isinstance(self.model, HybridPowerGNN):
                    if hasattr(batch, "graph_features"):
                        pred = self.model(batch, batch.graph_features)
                    else:
                        continue
                else:
                    pred = self.model(batch)

                loss = self.criterion(pred, batch.y)
                losses.append(loss.item())
                predictions.append(pred.cpu())
                targets.append(batch.y.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        # Calculate metrics
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        mape = (
            torch.mean(torch.abs((predictions - targets) / (targets + 1e-8))) * 100
        ).item()

        return {
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        }

    def predict(self, data: Data) -> float:
        """
        Make a prediction for a single graph.

        Args:
            data: PyG Data object

        Returns:
            Predicted power consumption

        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)

            if isinstance(self.model, HybridPowerGNN):
                if hasattr(data, "graph_features"):
                    pred = self.model(data, data.graph_features)
                else:
                    msg = "HybridPowerGNN requires graph_features"
                    raise ValueError(msg)
            else:
                pred = self.model(data)

            return pred.item()

    def save_model(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save the model

        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to load the model from

        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]


def load_dataset_from_json(
    design_names: list[str],
    feat_dir: Path,
    label_dir: Path,
    label_type: str = "Power",
) -> list[tuple[Data, float]]:
    """
    Load dataset from MasterRTL-style JSON files.

    Args:
        design_names: List of design names
        feat_dir: Directory containing feature JSON files
        label_dir: Directory containing label JSON files
        label_type: Type of label to load ('Power', 'Area', 'TNS', 'WNS')

    Returns:
        List of (data, label) tuples

    """
    dataset = []

    for design_name in design_names:
        # Load features
        feat_path = feat_dir / f"{design_name}_sog_vec_area.json"
        if not feat_path.exists():
            logger.warning("Feature file not found for %s", design_name)
            continue

        with feat_path.open() as f:
            features = json.load(f)

        # Load label
        label_path = label_dir / f"{design_name}.json"
        if not label_path.exists():
            logger.warning("Label file not found for %s", design_name)
            continue

        with label_path.open() as f:
            label_dict = json.load(f)

        if label_type not in label_dict:
            logger.warning("Label type %s not found for %s", label_type, design_name)
            continue

        label = label_dict[label_type]

        # Create a simple Data object with features
        # This is a placeholder - you'll need to load the actual graph structure
        x = torch.tensor([features], dtype=torch.float)
        data = Data(x=x)
        data.y = torch.tensor([label], dtype=torch.float)
        data.graph_features = torch.tensor(features, dtype=torch.float)

        dataset.append((data, label))

    return dataset
