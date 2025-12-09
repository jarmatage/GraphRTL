"""GNN models for RTL power estimation."""

import torch
import torch.nn.functional as nnf
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class PowerEstimationGNN(nn.Module):
    """
    Graph Neural Network for RTL power estimation.

    This model uses graph convolutions to learn representations of RTL designs
    and predicts power consumption at the graph level.

    Architecture:
        - Multiple GNN layers (GCN, GAT, GraphSAGE, or GIN)
        - Global pooling to get graph-level representation
        - MLP for final power prediction
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
        pooling: str = "mean",
    ) -> None:
        """
        Initialize the power estimation GNN.

        Args:
            in_channels: Number of input node features (default: 3 for bit_width,
            fanout, toggle_rate)
            hidden_channels: Number of hidden channels in GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution ('gcn', 'gat', 'sage', 'gin')
            pooling: Global pooling method ('mean', 'max', 'add')

        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.pooling = pooling

        # Create GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            if conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif conv_type == "gat":
                # GAT with 4 attention heads
                heads = 4 if i < num_layers - 1 else 1
                out_dim = (
                    hidden_channels // heads if i < num_layers - 1 else hidden_channels
                )
                self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
            elif conv_type == "sage":
                self.convs.append(SAGEConv(in_dim, hidden_channels))
            elif conv_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(mlp))
            else:
                msg = f"Unknown conv_type: {conv_type}"
                raise ValueError(msg)

            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # MLP for graph-level prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data: PyG Data object with x (node features) and edge_index

        Returns:
            Predicted power consumption (scalar)

        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Ensure Tensors
        if not isinstance(x, torch.Tensor):
            msg = f"Expected Tensor for x, got {type(x)}"
            raise TypeError(msg)

        # If batch is not provided, create it (single graph case)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = nnf.relu(x)
            x = nnf.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            msg = f"Unknown pooling: {self.pooling}"
            raise ValueError(msg)

        # MLP for final prediction
        x = self.mlp(x)

        return x.squeeze(-1)


class HybridPowerGNN(nn.Module):
    """
    Hybrid GNN that combines graph features with handcrafted features.

    This model combines:
    1. GNN-learned features from the graph structure
    2. Handcrafted graph-level features (like MasterRTL's approach)

    This often performs better than pure GNN or pure handcrafted features.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_graph_features: int = 14,  # From MasterRTL's cal_oper
        dropout: float = 0.2,
        conv_type: str = "gcn",
    ) -> None:
        """
        Initialize the hybrid power estimation model.

        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden channels in GNN layers
            num_layers: Number of GNN layers
            num_graph_features: Number of handcrafted graph-level features
            dropout: Dropout rate
            conv_type: Type of graph convolution

        """
        super().__init__()

        # GNN backbone
        self.gnn = PowerEstimationGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type,
        )

        # Remove the final MLP from GNN (we'll replace it with our own)
        self.gnn.mlp = nn.Identity()

        # Combined MLP
        combined_dim = hidden_channels + num_graph_features
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data: Data, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Args:
            data: PyG Data object with x (node features) and edge_index
            graph_features: Handcrafted graph-level features

        Returns:
            Predicted power consumption (scalar)

        """
        # Get GNN features
        gnn_features = self.gnn(data)

        # Ensure graph_features has the right shape
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)

        # Concatenate GNN and handcrafted features
        combined = torch.cat(
            [
                gnn_features.unsqueeze(0) if gnn_features.dim() == 0 else gnn_features,
                graph_features,
            ],
            dim=-1,
        )

        # Final prediction
        out = self.mlp(combined)

        return out.squeeze(-1)


class MultiTaskPowerGNN(nn.Module):
    """
    Multi-task GNN that predicts multiple PPA metrics simultaneously.

    Predicts:
        - Power consumption
        - Area
        - Timing (TNS, WNS)

    Multi-task learning can help the model learn better representations
    by sharing knowledge across related tasks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
    ) -> None:
        """
        Initialize the multi-task power estimation model.

        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden channels in GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution

        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # Shared GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels

            if conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif conv_type == "gat":
                heads = 4 if i < num_layers - 1 else 1
                out_dim = (
                    hidden_channels // heads if i < num_layers - 1 else hidden_channels
                )
                self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
            elif conv_type == "sage":
                self.convs.append(SAGEConv(in_dim, hidden_channels))
            elif conv_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(mlp))

            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Task-specific heads
        self.power_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.area_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.timing_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),  # TNS and WNS
        )

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task network.

        Args:
            data: PyG Data object with x (node features) and edge_index

        Returns:
            Dictionary with predictions for each task:
                - 'power': Power consumption
                - 'area': Area
                - 'timing': TNS and WNS

        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Ensure Tensors
        if not isinstance(x, torch.Tensor):
            msg = f"Expected Tensor for x, got {type(x)}"
            raise TypeError(msg)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Shared GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = nnf.relu(x)
            x = nnf.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Task-specific predictions
        power = self.power_head(x).squeeze(-1)
        area = self.area_head(x).squeeze(-1)
        timing = self.timing_head(x)  # [batch_size, 2] for TNS and WNS

        return {
            "power": power,
            "area": area,
            "tns": timing[:, 0],
            "wns": timing[:, 1],
        }
