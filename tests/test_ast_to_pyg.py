"""Test the AST to PyG converter."""

from pathlib import Path

import pytest

from graphrtl.sog.ast_to_pyg import ASTToPyG
from graphrtl.sog.directed_graph import DirectedGraph
from graphrtl.sog.sog_node import SOGNode


def test_directed_graph() -> None:
    """Test DirectedGraph basic functionality."""
    graph = DirectedGraph()
    node1 = SOGNode(name="test1", type="Wire", bit_width=8)
    node2 = SOGNode(name="test2", type="Reg", bit_width=16)

    graph.add_node("test1", node1)
    graph.add_node("test2", node2)
    graph.add_edge("test1", "test2")

    assert "test1" in graph.nodes
    assert "test2" in graph.nodes
    assert "test2" in graph.get_loads("test1")


def test_convert_verilog_file() -> None:
    """Test converting a Verilog file to PyG."""
    fixture_path = Path(__file__).parent / "fixtures" / "memory.sv"

    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    converter = ASTToPyG()
    data = converter.convert_verilog(fixture_path)

    # Check that we got a PyG Data object
    assert data is not None
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "node_names")
    assert hasattr(data, "node_types")

    # Check that we have nodes and edges
    assert data.x.shape[0] > 0  # At least one node
    assert data.x.shape[1] == 3  # 3 features per node
