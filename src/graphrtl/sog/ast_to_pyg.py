"""Convert Verilog AST from pyverilog to PyTorch Geometric graph."""

import re

import torch
from pyverilog.vparser.parser import parse
from torch_geometric.data import Data

from .directed_graph import DirectedGraph
from .sog_node import SOGNode


class ASTToPyG:
    """Convert Verilog AST to PyTorch Geometric graph."""

    def __init__(self, ast: object) -> None:
        """
        Initialize the converter with an AST.

        Args:
            ast: The AST from pyverilog.vparser.parser.parse

        """
        self.ast = ast
        self.graph = DirectedGraph()
        self.oper_label = 0
        self.const_label = 0
        self.wire_set: set[str] = set()
        self.wire_dict: dict[str, list[str]] = {}
        self.temp_dict: dict[str, list[str]] = {}

    def convert(self) -> Data:
        """
        Convert AST to PyTorch Geometric Data object.

        Returns:
            PyTorch Geometric Data object with the graph structure

        """
        # Step 1: Traverse AST and build graph
        self.traverse_ast(self.ast)

        # Step 2: Calculate node widths
        self._calculate_node_widths()

        # Step 3: Eliminate wire nodes
        self._eliminate_wires()

        # Step 4: Convert to PyG format
        return self._to_pyg()

    def traverse_ast(self, ast: object) -> None:
        """Traverse the AST and build the graph structure."""
        if ast is None:
            return

        node_type = ast.__class__.__name__

        # Handle declarations
        if node_type == "Decl":
            self._add_decl_node(ast)

        # Handle assignments
        if node_type in ["Assign", "NonblockingSubstitution", "BlockingSubstitution"]:
            self._add_assign(ast)

        # Handle control structures
        if node_type == "IfStatement":
            self._handle_if_statement(ast)

        if node_type in ["CaseStatement", "CasezStatement", "CasexStatement"]:
            self._handle_case_statement(ast)

        # Recursively traverse children
        if hasattr(ast, "children"):
            for child in ast.children():
                self.traverse_ast(child)

    def _add_decl_node(self, ast: object) -> None:
        """Add a declaration node to the graph."""
        if not hasattr(ast, "list") or not ast.list:
            return

        for item in ast.list:
            child_type = item.__class__.__name__
            name = str(item.name) if hasattr(item, "name") else None

            if name:
                width = self._get_width(item)
                node = SOGNode(name=name, type=child_type, bit_width=width)
                self.graph.add_node(name, node)

                if child_type == "Wire":
                    self.wire_set.add(name)

    def _add_assign(self, ast: object) -> str | None:
        """Handle assignment statements."""
        if not hasattr(ast, "left") or not hasattr(ast, "right"):
            return None

        lval = ast.left
        rval = ast.right

        # Get LHS name
        lhs_name = self._get_node_name(lval.var if hasattr(lval, "var") else lval)
        if not lhs_name:
            return None

        # Process RHS
        self._process_rvalue(rval.var if hasattr(rval, "var") else rval, lhs_name)

        return lhs_name

    def _create_operator_node(self, node_type: str, parent_name: str) -> str:
        """Create an operator node and return its name."""
        node_name = f"{node_type}{self.oper_label}"
        self.oper_label += 1
        node = SOGNode(name=node_name, type="Operator", bit_width=1)
        self.graph.add_node(node_name, node)
        self.graph.add_edge(parent_name, node_name)
        return node_name

    def _process_rvalue(self, ast: object, parent_name: str) -> None:  # noqa: C901
        """Process right-hand side of assignment."""
        if ast is None:
            return

        node_type = ast.__class__.__name__

        # Handle identifiers
        if node_type == "Identifier":
            node_name = str(ast.name)  # type: ignore[missing-attribute]
            if node_name not in self.graph.node_dict:
                node = SOGNode(name=node_name, type="Identifier", bit_width=1)
                self.graph.add_node(node_name, node)
            self.graph.add_edge(parent_name, node_name)
            return

        # Handle constants
        if node_type == "IntConst":
            node_name = f"Constant{self.const_label}"
            self.const_label += 1
            width = self._get_width_from_value(str(ast.value))  # type: ignore[missing-attribute]
            node = SOGNode(name=node_name, type="Constant", bit_width=width)
            self.graph.add_node(node_name, node)
            self.graph.add_edge(parent_name, node_name)
            return

        # Handle binary operators
        if node_type in [
            "Plus",
            "Minus",
            "Times",
            "Divide",
            "Mod",
            "And",
            "Or",
            "Xor",
            "Xnor",
            "Land",
            "Lor",
            "Eq",
            "NotEq",
            "LessThan",
            "GreaterThan",
            "LessEq",
            "GreaterEq",
            "Sll",
            "Srl",
            "Sra",
        ]:
            node_name = self._create_operator_node(node_type, parent_name)
            if hasattr(ast, "left"):
                self._process_rvalue(ast.left, node_name)
            if hasattr(ast, "right"):
                self._process_rvalue(ast.right, node_name)
            return

        # Handle unary operators
        if node_type in [
            "Unot",
            "Ulnot",
            "Uand",
            "Unand",
            "Uor",
            "Unor",
            "Uxor",
            "Uxnor",
            "Uplus",
            "Uminus",
        ]:
            node_name = self._create_operator_node(node_type, parent_name)
            if hasattr(ast, "right"):
                self._process_rvalue(ast.right, node_name)
            return

        # Handle concatenation
        if node_type == "Concat":
            node_name = f"Concat{self.oper_label}"
            self.oper_label += 1
            node = SOGNode(name=node_name, type="Concat", bit_width=1)
            self.graph.add_node(node_name, node)
            self.graph.add_edge(parent_name, node_name)
            if hasattr(ast, "list"):
                for child in ast.list:
                    self._process_rvalue(child, node_name)

    def _handle_if_statement(self, ast: object) -> None:
        """Handle if-else statements by creating mux nodes."""
        if not hasattr(ast, "cond"):
            return

        mux_name = f"Mux{self.oper_label}"
        self.oper_label += 1

        # Add mux node
        node = SOGNode(name=mux_name, type="Mux", bit_width=1)
        self.graph.add_node(mux_name, node)

        # Process condition
        self._process_rvalue(ast.cond, mux_name)

        # Process true and false branches
        if hasattr(ast, "true_statement") and ast.true_statement:
            self.traverse_ast(ast.true_statement)

        if hasattr(ast, "false_statement") and ast.false_statement:
            self.traverse_ast(ast.false_statement)

    def _handle_case_statement(self, ast: object) -> None:
        """Handle case statements."""
        if not hasattr(ast, "comp"):
            return

        mux_name = f"Mux{self.oper_label}"
        self.oper_label += 1

        node = SOGNode(name=mux_name, type="Mux", bit_width=1)
        self.graph.add_node(mux_name, node)

        # Process comparison value
        self._process_rvalue(ast.comp, mux_name)

        # Process cases
        if hasattr(ast, "caselist"):
            for case in ast.caselist:
                if hasattr(case, "statement"):
                    self.traverse_ast(case.statement)

    def _get_node_name(self, ast: object) -> str | None:
        """Extract node name from AST node."""
        if ast is None:
            return None

        node_type = ast.__class__.__name__

        if node_type == "Identifier" and hasattr(ast, "name"):
            return str(ast.name)

        if node_type == "Pointer" and hasattr(ast, "var"):
            var_name = str(ast.var.name) if hasattr(ast.var, "name") else None
            # Handle ptr which might be IntConst or Identifier
            if hasattr(ast, "ptr"):
                if hasattr(ast.ptr, "value"):
                    ptr = str(ast.ptr.value)
                elif hasattr(ast.ptr, "name"):
                    ptr = str(ast.ptr.name)
                else:
                    ptr = "0"
            else:
                ptr = "0"
            return f"{var_name}.PTR{ptr}" if var_name else None

        if node_type == "Partselect" and hasattr(ast, "var"):
            return str(ast.var.name) if hasattr(ast.var, "name") else None

        return None

    def _get_width(self, ast: object) -> int:
        """Calculate bit width from AST node."""
        if not hasattr(ast, "width"):
            return 1

        width = ast.width
        if width is None:
            return 1

        if hasattr(width, "msb") and hasattr(width, "lsb"):
            try:
                msb = int(width.msb.value) if hasattr(width.msb, "value") else 0
                lsb = int(width.lsb.value) if hasattr(width.lsb, "value") else 0
                return abs(msb - lsb) + 1
            except (ValueError, AttributeError):
                return 1

        return 1

    def _get_width_from_value(self, value: str) -> int:
        """Get bit width from a constant value string."""
        # Handle Verilog number format: <width>'<base><value>
        match = re.match(r"(\d+)'([bBoOdDhH])", value)
        if match:
            return int(match.group(1))

        # Handle simple 0 or 1
        if value in ["0", "1"]:
            return 1

        # For other cases, estimate based on string length
        return max(1, len(value))

    def _calculate_node_widths(self) -> None:
        """Calculate and propagate bit widths through the graph."""
        # Simple width calculation - can be enhanced
        for name, node in self.graph.node_dict.items():
            if node.bit_width == 1 and name in self.graph.edges:
                # Try to infer width from connected nodes
                neighbors = self.graph.get_neighbors(name)
                if neighbors:
                    max_width = max(
                        (
                            self.graph.node_dict[n].bit_width
                            for n in neighbors
                            if n in self.graph.node_dict
                        ),
                        default=1,
                    )
                    node.bit_width = max_width

    def _eliminate_wires(self) -> None:
        """Remove wire nodes by connecting their inputs directly to outputs."""
        # Identify all wire nodes
        wire_nodes = self.wire_set.copy()

        # Iteratively eliminate wires
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for wire in list(wire_nodes):
                if wire not in self.graph.node_dict:
                    wire_nodes.remove(wire)
                    continue

                # Get wire's inputs and outputs
                wire_outputs = self.graph.get_neighbors(wire)

                # Find nodes that have this wire as output
                wire_inputs = [
                    node
                    for node in self.graph.get_all_nodes()
                    if wire in self.graph.get_neighbors(node)
                ]

                if not wire_inputs or not wire_outputs:
                    # Remove isolated wire
                    self.graph.remove_node(wire)
                    wire_nodes.remove(wire)
                    changed = True
                    continue

                # Connect inputs directly to outputs
                for in_node in wire_inputs:
                    for out_node in wire_outputs:
                        self.graph.add_edge(in_node, out_node)

                # Remove the wire node
                self.graph.remove_node(wire)
                wire_nodes.remove(wire)
                changed = True

    def _to_pyg(self) -> Data:
        """Convert the graph to PyTorch Geometric Data format."""
        # Create node mapping
        nodes = sorted(self.graph.get_all_nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Create edge index
        edge_list = [
            [node_to_idx[from_node], node_to_idx[to_node]]
            for from_node in nodes
            for to_node in self.graph.get_neighbors(from_node)
            if to_node in node_to_idx
        ]

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create node features
        node_features = [
            [
                float(self.graph.node_dict[node_name].bit_width),
                float(self.graph.node_dict[node_name].fanout),
                float(self.graph.node_dict[node_name].toggle_rate),
            ]
            for node_name in nodes
        ]

        x = torch.tensor(node_features, dtype=torch.float)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        # Store additional metadata
        data.node_names = nodes
        data.node_types = [self.graph.node_dict[n].type for n in nodes]

        return data


def convert_verilog_to_pyg(verilog_file: str) -> Data:
    """
    Convert a Verilog file to PyG graph.

    Args:
        verilog_file: Path to the Verilog file

    Returns:
        PyTorch Geometric Data object

    """
    ast, _ = parse([verilog_file], outputdir="/tmp/ply")  # noqa: S108
    converter = ASTToPyG(ast)
    return converter.convert()
