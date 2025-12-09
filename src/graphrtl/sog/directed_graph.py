"""Simple directed graph implementation for SOG."""

from .sog_node import SOGNode


class DirectedGraph:
    """Simple directed graph for intermediate representation."""

    def __init__(self) -> None:
        """Initialize an empty directed graph."""
        self.edges: dict[str, list[str]] = {}
        self.node_dict: dict[str, SOGNode] = {}

    def add_node(self, name: str, node: SOGNode) -> None:
        """Add a node to the graph."""
        self.node_dict[name] = node
        if name not in self.edges:
            self.edges[name] = []

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge from one node to another."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        if to_node not in self.edges:
            self.edges[to_node] = []
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

    def get_neighbors(self, node: str) -> list[str]:
        """Get all neighbors (outgoing edges) of a node."""
        return self.edges.get(node, [])

    def get_all_nodes(self) -> list[str]:
        """Get all node names."""
        return list(self.node_dict.keys())

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph."""
        if node in self.node_dict:
            del self.node_dict[node]
        if node in self.edges:
            del self.edges[node]
        # Remove edges pointing to this node
        for neighbors in self.edges.values():
            if node in neighbors:
                neighbors.remove(node)
