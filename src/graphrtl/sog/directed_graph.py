"""Simple directed graph implementation for SOG."""

from collections import defaultdict

from .sog_node import SOGNode


class DirectedGraph:
    """Simple directed graph for intermediate representation."""

    def __init__(self) -> None:
        """Initialize an empty directed graph."""
        self._nodes: dict[str, SOGNode] = {}
        self._edges: dict[str, set[str]] = defaultdict(set[str])

    @property
    def nodes(self) -> dict[str, SOGNode]:
        """Get the nodes of the graph."""
        return self._nodes

    @property
    def edges(self) -> dict[str, set[str]]:
        """Get the edges of the graph."""
        return self._edges

    def add_node(self, name: str, node: SOGNode) -> None:
        """Add a node to the graph."""
        self.nodes[name] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge from one node to another."""
        self.edges[from_node].add(to_node)

    def get_loads(self, node: str) -> set[str]:
        """Get all the loads of a given node."""
        return self.edges[node]

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph."""
        self.nodes.pop(node, None)
        self.edges.pop(node, None)
        for neighbors in self.edges.values():
            neighbors.discard(node)
