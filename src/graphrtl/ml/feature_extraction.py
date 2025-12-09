"""Feature extraction utilities for RTL graphs inspired by MasterRTL."""

import re

import torch
from torch_geometric.data import Data

from graphrtl.sog.directed_graph import DirectedGraph

# Standard data for analytical estimation (from MasterRTL)
STD_DATA = {
    "area_seq": {
        "DFF": 4.522,
        "Reg": 4.522,
    },
    "area_comb": {
        "And": 1.064,
        "Or": 1.064,
        "Xor": 1.596,
        "Xnor": 1.596,
        "Not": 0.798,
        "Unot": 0.798,
        "Cond": 1.862,
        "Mux": 1.862,
        "Plus": 4.256,
        "Minus": 4.256,
        "Times": 37.5,
        "Divide": 40,
        "Mod": 35,
        "LessThan": 35,
        "GreaterThan": 35,
    },
    "stat_pwr": {
        "DFF": 0.07911,
        "Reg": 0.07911,
        "And": 0.02507,
        "Or": 0.02269,
        "Xor": 0.03616,
        "Xnor": 0.03616,
        "Not": 0.106,
        "Unot": 0.106,
        "Cond": 0.03593,
        "Mux": 0.03593,
        "Plus": 0.07576,
        "Minus": 0.07576,
        "Times": 0.1,
        "Divide": 0.1,
        "Mod": 0.1,
        "LessThan": 0.3,
        "GreaterThan": 0.3,
    },
    "dyn_pwr": {
        "DFF": 3.98,
        "Reg": 3.98,
        "And": 5.2,
        "Or": 2.71,
        "Xor": 2.43,
        "Xnor": 2.43,
        "Not": 4.61,
        "Unot": 4.61,
        "Cond": 3.38,
        "Mux": 3.38,
        "Plus": 4.95,
        "Minus": 4.95,
        "Times": 37.5,
        "Divide": 40,
        "Mod": 35,
    },
}


def extract_graph_features(graph: DirectedGraph) -> list[float]:  # noqa: C901, PLR0912
    """
    Extract handcrafted graph-level features from DirectedGraph.

    This follows MasterRTL's feature extraction approach from cal_oper().

    Args:
        graph: DirectedGraph object from GraphRTL

    Returns:
        List of 14 features:
            [0] seq_num: Number of sequential element bits
            [1] fanout_sum: Sum of fanouts
            [2] io_num: Number of I/O bits
            [3] and_num: Number of AND gates
            [4] or_num: Number of OR gates
            [5] not_num: Number of NOT gates
            [6] xor_num: Number of XOR gates
            [7] mux_num: Number of MUX gates
            [8] seq_area: Estimated sequential area
            [9] comb_area: Estimated combinational area
            [10] total_area: Total estimated area
            [11] stat_pwr: Estimated static power
            [12] dyn_pwr: Estimated dynamic power
            [13] total_pwr: Total estimated power

    """
    seq_num, comb_num, io_num, fanout_sum = 0, 0, 0, 0
    and_num, or_num, not_num, xor_num, mux_num = 0, 0, 0, 0, 0
    seq_area, comb_area, stat_pwr, dyn_pwr = 0.0, 0.0, 0.0, 0.0

    for name, node in graph.nodes.items():
        node_type = node.type
        width = node.width if hasattr(node, "width") else node.bit_width

        # Sequential elements
        if node_type in ["Reg", "Register"]:
            fanout_sum += len(graph.edges[name])
            seq_num += width
            seq_area += STD_DATA["area_seq"].get("Reg", 4.522) * width
            stat_pwr += STD_DATA["stat_pwr"].get("Reg", 0.07911) * width
            dyn_pwr += STD_DATA["dyn_pwr"].get("Reg", 3.98) * width

        # Combinational logic
        elif node_type in ["Operator", "UnaryOperator", "Mux", "Cond"]:
            comb_num += width

            # Extract operator type from name (e.g., "And123" -> "And")
            op_match = re.match(r"([A-Z][a-z]*)\d*", name)
            if op_match:
                op = op_match.group(1)
            elif node_type in ["Mux", "Cond"]:
                op = node_type
            else:
                op = "And"  # Default

            # Add area and power estimates
            if op in STD_DATA["area_comb"]:
                comb_area += STD_DATA["area_comb"][op] * width
            if op in STD_DATA["stat_pwr"]:
                stat_pwr += STD_DATA["stat_pwr"][op] * width
            if op in STD_DATA["dyn_pwr"]:
                dyn_pwr += STD_DATA["dyn_pwr"][op] * width

            # Count gate types
            if "And" in op:
                and_num += 1
            elif "Or" in op:
                or_num += 1
            elif "not" in op.lower():
                not_num += 1
            elif "Xor" in op:
                xor_num += 1
            elif op in ["Cond", "Mux"]:
                mux_num += 1

        # I/O ports
        elif node_type in ["Output", "Input", "Inout"]:
            io_num += width

    # Totals
    seq_area = round(seq_area, 0)
    comb_area = round(comb_area, 0)
    total_area = seq_area + comb_area
    stat_pwr = round(stat_pwr, 0)
    dyn_pwr = round(dyn_pwr, 0)
    total_pwr = stat_pwr + dyn_pwr

    return [
        float(seq_num),
        float(fanout_sum),
        float(io_num),
        float(and_num),
        float(or_num),
        float(not_num),
        float(xor_num),
        float(mux_num),
        float(seq_area),
        float(comb_area),
        float(total_area),
        float(stat_pwr),
        float(dyn_pwr),
        float(total_pwr),
    ]


def extract_features_from_pyg(data: Data) -> list[float]:
    """
    Extract handcrafted features from PyG Data object.

    This is a simpler version when you only have the PyG graph.

    Args:
        data: PyG Data object

    Returns:
        List of graph-level features

    """
    if not isinstance(data.x, torch.Tensor):
        msg = f"Expected Tensor, got {type(data.x)}"
        raise TypeError(msg)
    if not isinstance(data.edge_index, torch.Tensor):
        msg = f"Expected Tensor, got {type(data.edge_index)}"
        raise TypeError(msg)

    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    # Calculate graph statistics
    avg_degree = num_edges / max(num_nodes, 1)

    # Calculate node feature statistics
    bit_widths = data.x[:, 0] if data.x.shape[1] > 0 else torch.zeros(num_nodes)
    fanouts = data.x[:, 1] if data.x.shape[1] > 1 else torch.zeros(num_nodes)
    toggle_rates = data.x[:, 2] if data.x.shape[1] > 2 else torch.zeros(num_nodes)  # noqa: PLR2004

    return [
        float(num_nodes),
        float(num_edges),
        float(avg_degree),
        float(bit_widths.sum()),
        float(bit_widths.mean()),
        float(bit_widths.std()),
        float(fanouts.sum()),
        float(fanouts.mean()),
        float(toggle_rates.sum()),
        float(toggle_rates.mean()),
    ]


def add_graph_features_to_data(data: Data, graph: DirectedGraph | None = None) -> Data:
    """
    Add graph-level features as an attribute to PyG Data object.

    Args:
        data: PyG Data object
        graph: Optional DirectedGraph for more detailed features

    Returns:
        Updated Data object with graph_features attribute

    """
    if graph is not None:
        features = extract_graph_features(graph)
    else:
        features = extract_features_from_pyg(data)

    data.graph_features = torch.tensor(features, dtype=torch.float)
    return data


def calculate_analytical_power(
    graph: DirectedGraph, toggle_rates: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Calculate analytical power estimation similar to MasterRTL.

    Args:
        graph: DirectedGraph object
        toggle_rates: Optional dictionary of node names to toggle rates

    Returns:
        Dictionary with power estimates:
            - static_power: Static power consumption
            - dynamic_power: Dynamic power consumption
            - total_power: Total power consumption

    """
    stat_pwr = 0.0
    dyn_pwr = 0.0

    for name, node in graph.nodes.items():
        node_type = node.type
        width = node.width if hasattr(node, "width") else node.bit_width

        # Get toggle rate (if provided)
        tr = toggle_rates.get(name, 0.1) if toggle_rates else 0.1

        # Calculate power based on type
        if node_type in ["Reg", "Register"]:
            stat_pwr += STD_DATA["stat_pwr"].get("Reg", 0.07911) * width
            dyn_pwr += STD_DATA["dyn_pwr"].get("Reg", 3.98) * width * tr

        elif node_type in ["Operator", "UnaryOperator", "Mux", "Cond"]:
            op_match = re.match(r"([A-Z][a-z]*)\d*", name)
            op = op_match.group(1) if op_match else node_type

            if op in STD_DATA["stat_pwr"]:
                stat_pwr += STD_DATA["stat_pwr"][op] * width
            if op in STD_DATA["dyn_pwr"]:
                dyn_pwr += STD_DATA["dyn_pwr"][op] * width * tr

    return {
        "static_power": round(stat_pwr, 4),
        "dynamic_power": round(dyn_pwr, 4),
        "total_power": round(stat_pwr + dyn_pwr, 4),
    }
