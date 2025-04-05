# jeanplot/network_diagram.py
from typing import Dict, List, Optional, Any, Tuple, Set
from pydantic import Field, PrivateAttr
import pandas as pd
from collections import defaultdict
import numpy as np
import logging
import re  # For parsing ERN names

from .component import Component
from .container import Container, Overlay
from .models import Size, BoxStyle, LayoutConstraints, Offset
from .text import Text
from .svg import SVGElement  # Keep import
from .connector import (
    Connection,
    StraightCurve,
    LineEndArrow,
    SimpleBezierCurve,
)  # Need Bezier for ERN internal
from .network_utils import get_tu_informations, TUInfo
from .style import jstyle
from .debug import debug_print

logger = logging.getLogger(__name__)


class NodeComponent(Container):
    """visual representation for a graph node."""

    node_id: Any
    node_type: str = "unknown"
    node_label: Optional[str] = None  # Text component created based on this

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        # Add text child only if a label is provided and it's not a container type meant to hold other specific visuals
        if self.node_label and self.node_type not in ["aggregation", "source", "ern", "inverse"]:
            self.add_child(Text(text=self.node_label, id=f"lbl_{self.id}"))
        elif self.node_type == "inverse":  # Special case for "Inv" label
            self.add_child(Text(text="Inv", id=f"lbl_{self.id}"))


class TUComponent(Container):
    """visual representation for a TU (simple thick line)."""

    tu_id: str
    style_class: list[str] = ["tu-component"]
    # Style set in theme to be a black bar


class NetworkDiagram(Container):
    network: Any
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row",  # Layers arranged horizontally
            align_items="center",  # Vertically center layers relative to each other
            justify_content="start",  # Align layers to the start
            gap=80,  # Increased gap between layer columns
        )
    )
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(background_color="#ffffff", padding=(30, 30, 30, 30))
    )

    # internal state
    _compute_graph: pd.DataFrame = PrivateAttr(default=None)
    _tu_infos: Dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    _layers: List[List[Any]] = PrivateAttr(default_factory=list)
    _node_map: Dict[Any, Component] = PrivateAttr(
        default_factory=dict
    )  # node_id | f"{fwd_id}_inv_viz" -> Component
    _processed_nodes: Set[Any] = PrivateAttr(default_factory=set)
    _inverse_chains: Dict[Any, Set[Any]] = PrivateAttr(default_factory=lambda: defaultdict(set))
    _inv_node_to_root: Dict[Any, Any] = PrivateAttr(default_factory=dict)
    _root_has_inv_viz: Set[Any] = PrivateAttr(default_factory=set)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.network:
            self._compute_graph = self.network.compute_graph
            self._tu_infos = get_tu_informations(self.network)
            self._prepare_layout()

    def _log_debug(self, message: str, data=None):
        debug_print(self.id or "NetworkDiagram", message, data)

    def _process_inverse_nodes(self):
        """Identify chains of inverse nodes and map them to their forward root."""
        self._inverse_chains.clear()
        self._inv_node_to_root.clear()
        graph = self._compute_graph

        all_inverse_nodes = {
            idx for idx, row in graph.iterrows() if row.get("is_inverse_of") is not None
        }
        processed_inv_nodes = set()

        for inv_node_id in all_inverse_nodes:
            if inv_node_id in processed_inv_nodes:
                continue

            current_chain = set()
            queue = [inv_node_id]
            chain_root_fwd_id = None
            visited_in_chain = set()

            while queue:
                curr_inv_id = queue.pop(0)
                if curr_inv_id in processed_inv_nodes or curr_inv_id in visited_in_chain:
                    continue

                visited_in_chain.add(curr_inv_id)

                # Check if this node is actually in the graph index before proceeding
                if curr_inv_id not in graph.index:
                    logger.warning(f"Inverse node {curr_inv_id} referenced but not found in graph.")
                    continue

                current_chain.add(curr_inv_id)
                temp_root = curr_inv_id
                traversed_up = {temp_root}  # Detect cycles in upward traversal
                is_valid_chain = True
                while True:
                    fwd_id = graph.loc[temp_root].get("is_inverse_of")
                    if (
                        fwd_id is None
                    ):  # This node itself is the forward root (shouldn't happen for starting node)
                        chain_root_fwd_id = temp_root
                        break
                    elif fwd_id in traversed_up:  # Cycle detected
                        logger.warning(f"Cycle detected in inverse chain involving {fwd_id}")
                        is_valid_chain = False
                        break
                    elif fwd_id not in all_inverse_nodes:  # Found the actual forward root
                        chain_root_fwd_id = fwd_id
                        break
                    else:  # Continue up the chain
                        # Check if the next node exists before accessing it
                        if fwd_id not in graph.index:
                            logger.warning(
                                f"Inverse node {fwd_id} referenced by {temp_root} not found."
                            )
                            is_valid_chain = False
                            break
                        temp_root = fwd_id
                        traversed_up.add(temp_root)

                if not is_valid_chain:
                    chain_root_fwd_id = None  # Invalidate this chain root finding attempt
                    break  # Stop processing this specific branch from queue item

                # Only add parents if we successfully found a root for the current node
                if chain_root_fwd_id is not None:
                    # Look for parent inverse nodes
                    inputs = graph.loc[curr_inv_id].get("input_from", [])
                    for parent_id, _ in inputs:
                        if (
                            parent_id in all_inverse_nodes
                            and parent_id not in processed_inv_nodes
                            and parent_id not in visited_in_chain
                        ):
                            queue.append(parent_id)

            # Mark all nodes in the successfully traced chain (if valid root found)
            if chain_root_fwd_id is not None:
                self._inverse_chains[chain_root_fwd_id].update(current_chain)
                for node in current_chain:
                    self._inv_node_to_root[node] = chain_root_fwd_id
                processed_inv_nodes.update(current_chain)
            else:
                # Mark nodes in the invalid/failed chain attempt as processed anyway
                processed_inv_nodes.update(current_chain)

        self._log_debug("Inverse Node Roots Mapping", self._inv_node_to_root)
        # self._log_debug("Inverse Chains Per Root", self._inverse_chains) # Can be verbose

    def _prepare_layout(self):
        """Filters nodes, performs topological sort, and processes inverses."""
        if self._compute_graph is None or self._compute_graph.empty:
            return

        self._process_inverse_nodes()  # Find inverse chains first

        non_inverse_nodes = [
            idx for idx, row in self._compute_graph.iterrows() if idx not in self._inv_node_to_root
        ]

        try:
            if hasattr(self.network, "topological_order"):
                topo_order_result = self.network.topological_order(non_inverse_nodes)
                if topo_order_result and isinstance(topo_order_result[0], list):
                    self._layers = topo_order_result
                elif topo_order_result:
                    self._layers = [[n] for n in topo_order_result]
                else:  # Handle empty result
                    self._layers = [non_inverse_nodes] if non_inverse_nodes else []
            else:
                # Fallback - group by type (very approximate)
                logger.warning(
                    "Network object missing topological_order method. Using approximate layout."
                )
                grouped = defaultdict(list)
                type_order = [
                    "input",
                    "aggregation",
                    "source",
                    "transcription",
                    "translation",
                    "sequestron_ERN",
                    "output",
                ]  # Define a rough order
                for node_id in non_inverse_nodes:
                    grouped[self._compute_graph.loc[node_id, "type"]].append(node_id)
                self._layers = [grouped[t] for t in type_order if t in grouped]
                # Add any remaining types
                remaining_nodes = [
                    nodes for type_key, nodes in grouped.items() if type_key not in type_order
                ]
                self._layers.extend(remaining_nodes)

        except Exception as e:
            logger.error(f"Topological sort failed: {e}. Using single layer fallback.")
            self._layers = [non_inverse_nodes]  # Single layer fallback

        self._log_debug("Layout Layers (non-inverse)", self._layers)

    def _get_node_label(self, node_id, node_type, row):
        """Determine the display label for a node."""
        if node_type == "transcription":
            return "Tx"
        if node_type == "translation":
            return "Tl"
        if node_type == "inverse":
            return "Inv"  # Should be handled by specific Inv component creation

        # ERN: Extract name like "ERN A" from seq_name "ERN::Csy4#Csy4_rec"
        if node_type == "sequestron_ERN":
            seq_name = row.get("extra", {}).get("seq_name", "")
            match = re.match(r"ERN::([^#]+)", seq_name)
            # Simple naming for now, could be improved
            ern_letter = chr(
                ord("A")
                + list(
                    self._compute_graph[self._compute_graph["type"] == "sequestron_ERN"].index
                ).index(node_id)
            )
            return f"ERN {ern_letter}"

        # Output: Use marker name if possible
        if node_type == "output":
            marker_name = None
            cdg_input_list = row.get("cdg_input", [])
            if cdg_input_list:
                first_input_idx = cdg_input_list[0]
                if first_input_idx in self.network.central_dogma_graph.index:
                    content = self.network.central_dogma_graph.loc[first_input_idx].get("content")
                    if content and isinstance(content, tuple):
                        marker_name = content[0]
            return marker_name or "Y"  # Default to Y

        # Input: Try to find name via inverse path or use ID
        if node_type == "input":
            # Basic name for now, finding original source name can be complex
            extra = row.get("extra", {})
            input_pos = extra.get("input_position")
            input_label = f"In_{node_id}"  # Default fallback
            if input_pos is not None:
                # Try a more generic input name if possible
                input_label = f"X{input_pos + 1}"  # Like X1, X2 etc.
                # Could potentially trace back inverse path to find a better name from source/agg
            return input_label

        # Aggregation/Source: Label handled separately (placed below/beside)
        if node_type in ["aggregation", "source"]:
            return None  # Don't put label inside container

        # Default: Use the node ID as string
        return str(node_id)

    def _get_marker_class(self, node_id, row) -> Optional[str]:
        """Get the marker style class (e.g., 'marker-EBFP2') if applicable."""
        marker_name = None
        node_type = row.get("type")

        # Input/Output nodes: check connected TUs via inverse path or cdg_input
        if node_type in ["input", "output"]:
            # For output, check direct cdg_input
            cdg_list = row.get("cdg_input", [])
            # For input, need to trace back the inverse path (complex, skip for now or use simple method)
            # Simplified: Check if marker info was stored in 'extra' during graph creation? (unlikely)

            if cdg_list:
                cdg_idx = cdg_list[0]
                if cdg_idx in self.network.central_dogma_graph.index:
                    tu_ids = self.network.central_dogma_graph.loc[cdg_idx].get("tu_id", [])
                    if tu_ids:
                        tu_id = tu_ids[0]
                        if tu_id in self._tu_infos:
                            marker_name = self._tu_infos[tu_id].cotx_marker

        # Aggregation/Source nodes: Check TUs listed in cdg_output
        elif node_type in ["aggregation", "source"]:
            cdg_outputs = row.get("cdg_output", [])
            outputs_list = cdg_outputs if isinstance(cdg_outputs, list) else [cdg_outputs]
            for cdg_idx in outputs_list:
                if cdg_idx in self.network.central_dogma_graph.index:
                    tu_ids = self.network.central_dogma_graph.loc[cdg_idx].get("tu_id", [])
                    if tu_ids:
                        tu_id = tu_ids[0]
                        if tu_id in self._tu_infos:
                            marker_name = self._tu_infos[tu_id].cotx_marker
                            if marker_name:
                                break  # Found first marker

        if marker_name:
            # Sanitize and convert to upper for class name consistency
            safe_marker_name = re.sub(r"[^A-Z0-9_]", "", marker_name.upper())
            return f"marker-{safe_marker_name}"
        return None

    def _create_node_component(self, node_id: Any) -> Optional[Component]:
        """creates the appropriate jeanplot component for a given node_id."""
        if node_id not in self._compute_graph.index:
            logger.warning(f"Node ID {node_id} not found in compute graph.")
            return None

        row = self._compute_graph.loc[node_id]
        node_type = row.get("type", "unknown")
        comp_id = f"node_{node_id}"
        node_label = self._get_node_label(node_id, node_type, row)

        # Base component creation
        comp = NodeComponent(
            id=comp_id, node_id=node_id, node_type=node_type, node_label=node_label
        )

        # Add marker class if applicable
        marker_class = self._get_marker_class(node_id, row)
        if marker_class:
            comp.style_class.append(marker_class)

        # --- Type-specific modifications ---
        if node_type == "sequestron_ERN":
            # ERN needs internal Tx/Tl and external label
            comp.children = []  # Clear default label if any was added
            tx_node = NodeComponent(
                id=f"{comp_id}_tx", node_id=None, node_type="transcription", node_label="Tx"
            )
            tl_node = NodeComponent(
                id=f"{comp_id}_tl", node_id=None, node_type="translation", node_label="Tl"
            )
            comp.add_children([tx_node, tl_node])

            # Add ERN name label above
            ern_label_text = self._get_node_label(node_id, node_type, row)  # Get the "ERN A" label
            if ern_label_text:
                ern_label = Text(
                    text=ern_label_text, id=f"lbl_{comp_id}", font_size=8, color="#555"
                )
                # Use overlay for label placement above the ERN container
                label_overlay = Container(
                    id=f"lbl_cont_{comp_id}",
                    is_overlay=True,
                    children=[ern_label],
                    offset=Offset(relative=(0.5, 0), absolute=(0, -10)),  # Center above
                    parent=comp,  # Set parent link manually for overlay offset calculation
                )
                comp.add_child(label_overlay)

            # TODO: Add internal red regulation line (requires custom path drawing)

        elif node_type in ["aggregation", "source"]:
            comp.children = []  # Clear default label
            # Add TU representations
            tu_ids = set()
            cdg_outputs = row.get("cdg_output", [])
            outputs_list = cdg_outputs if isinstance(cdg_outputs, list) else [cdg_outputs]
            for cdg_idx in outputs_list:
                if cdg_idx in self.network.central_dogma_graph.index:
                    tu_id_list = self.network.central_dogma_graph.loc[cdg_idx].get("tu_id")
                    if tu_id_list and isinstance(tu_id_list, list):
                        tu_ids.add(tu_id_list[0])

            for tu_id in sorted(list(tu_ids)):
                comp.add_child(TUComponent(tu_id=tu_id, id=f"tu_{comp_id}_{tu_id}"))

            # Add Agg/Source ID label below (optional, based on 'extra')
            extra_info = row.get("extra", {})
            extra_id = extra_info.get("id")
            # Example: Add ratio label if present
            ratios = extra_info.get("ratios")
            if ratios:
                ratio_label_str = ":".join(
                    map(lambda x: f"{x:.1f}" if isinstance(x, float) else str(x), ratios)
                )
                ratio_label = Text(
                    text=ratio_label_str, id=f"ratio_{comp_id}", font_size=6, color="#666"
                )
                label_offset = Offset(relative=(0.5, 1), absolute=(0, 5))  # Position below
                # Add label as overlay
                ratio_label_cont = Container(
                    is_overlay=True, children=[ratio_label], offset=label_offset, parent=comp
                )
                comp.add_child(ratio_label_cont)

        comp.style_class.insert(0, "node-component")  # Ensure base class is first
        return comp

    def _create_inverse_component(self, forward_node_id: Any) -> Optional[Component]:
        """creates the collapsed 'Inv' visual representation."""
        inv_comp_id = f"node_{forward_node_id}_inv_viz"  # Specific ID for visual node
        # node_label is handled by NodeComponent based on node_type='inverse'
        inv_node = NodeComponent(
            id=inv_comp_id,
            node_id=forward_node_id,  # Logically linked
            node_type="inverse",
            node_label="Inv",  # Explicit label passed to ensure it's created
        )
        inv_node.style_class.insert(0, "node-component")
        return inv_node

    def build_diagram(self):
        """creates and arranges all components based on the network."""
        self.children = []
        self._node_map.clear()
        self._processed_nodes.clear()
        self._root_has_inv_viz.clear()

        if not self._layers:
            self._log_debug("No layers to build diagram.")
            return

        # create components layer by layer
        for i, layer_nodes in enumerate(self._layers):
            layer_container = Container(
                id=f"layer_{i}",
                style_class=["diagram-layer"],
                layout=LayoutConstraints(
                    direction="column",  # Nodes arranged vertically
                    align_items="center",  # Center nodes horizontally in the column
                    justify_content="space-around",  # Distribute nodes vertically
                    gap=25,  # Increased gap between nodes in a layer
                ),
            )

            added_to_layer = 0
            for node_id in layer_nodes:
                # Skip inverse nodes, they are handled by their forward root
                if node_id in self._inv_node_to_root:
                    continue

                comp = self._create_node_component(node_id)
                if comp:
                    layer_container.add_child(comp)
                    self._node_map[node_id] = comp
                    self._processed_nodes.add(node_id)
                    added_to_layer += 1

                    # Check if this node is the root of any inverse chains
                    if node_id in self._inverse_chains and node_id not in self._root_has_inv_viz:
                        inv_comp = self._create_inverse_component(node_id)
                        if inv_comp:
                            # --- Inverse Node Placement ---
                            # Place inverse node slightly below and to the right of the forward node
                            fwd_height = (
                                comp._dimensions.height if hasattr(comp, "_dimensions") else 25
                            )  # Use default if not measured yet
                            inv_offset_x = (
                                comp.min_dimensions.width + 30
                            )  # Horizontal offset based on fwd node size
                            inv_offset_y = fwd_height * 0.3  # Small vertical offset downwards

                            inv_comp.offset = Offset(absolute=(inv_offset_x, inv_offset_y))
                            # Add as an overlay relative to the forward component's container (the layer)
                            inv_comp.parent = layer_container  # Set parent for offset calculation
                            inv_comp.is_overlay = True  # Render on top

                            layer_container.add_child(
                                inv_comp
                            )  # Add to same layer container but as overlay
                            self._node_map[f"{node_id}_inv_viz"] = inv_comp
                            self._root_has_inv_viz.add(node_id)

            if added_to_layer > 0:  # Only add layer if it has non-inverse nodes
                self.add_child(layer_container)

        # create connections (must happen AFTER all nodes are created and potentially measured)
        connections = []
        processed_edges = set()

        # --- Connection Pass 1: Regular graph edges ---
        for src_node_id, row in self._compute_graph.iterrows():
            outputs = row.get("output_to", [])
            for tgt_node_id, _ in outputs:
                if tgt_node_id not in self._compute_graph.index:
                    continue

                # Determine the visual source component
                start_comp_key = (
                    f"{self._inv_node_to_root[src_node_id]}_inv_viz"
                    if src_node_id in self._inv_node_to_root
                    else src_node_id
                )
                start_comp = self._node_map.get(start_comp_key)

                # Determine the visual target component
                end_comp_key = (
                    f"{self._inv_node_to_root[tgt_node_id]}_inv_viz"
                    if tgt_node_id in self._inv_node_to_root
                    else tgt_node_id
                )
                end_comp = self._node_map.get(end_comp_key)

                # Skip connections entirely between two "Inv" visualization nodes
                if "_inv_viz" in str(start_comp_key) and "_inv_viz" in str(end_comp_key):
                    continue

                if start_comp and end_comp:
                    edge_tuple = (start_comp.id, end_comp.id)  # Use visual component IDs
                    if edge_tuple not in processed_edges:
                        conn = Connection(
                            id=f"conn_{src_node_id}_{tgt_node_id}",
                            start_component=start_comp,
                            end_component=end_comp,
                            curve_type=StraightCurve(),
                            style_class=["diagram-connection"],
                            is_overlay=True,
                        )
                        # Adjust offsets for specific container types if needed
                        if isinstance(start_comp, NodeComponent) and start_comp.node_type in [
                            "aggregation",
                            "source",
                        ]:
                            conn.start_offset = Offset(
                                relative=(1.0, 0.5)
                            )  # Connect from right-middle
                        if isinstance(end_comp, NodeComponent) and end_comp.node_type in [
                            "aggregation",
                            "source",
                        ]:
                            conn.end_offset = Offset(relative=(0.0, 0.5))  # Connect to left-middle

                        connections.append(conn)
                        processed_edges.add(edge_tuple)

        # --- Connection Pass 2: Dashed lines to Inv nodes ---
        for fwd_id in self._root_has_inv_viz:
            fwd_comp = self._node_map.get(fwd_id)
            inv_viz_comp = self._node_map.get(f"{fwd_id}_inv_viz")
            if fwd_comp and inv_viz_comp:
                inv_conn = Connection(
                    id=f"conn_inv_{fwd_id}",
                    start_component=fwd_comp,
                    end_component=inv_viz_comp,
                    curve_type=StraightCurve(),
                    style_class=["diagram-inverse-connection"],
                    is_overlay=True,
                )
                connections.append(inv_conn)

        self.add_children(connections)  # Add all connections at the end
        self._log_debug(
            f"Built diagram with {len(self._node_map)} visual nodes and {len(connections)} connections."
        )

    def measure_and_layout(self, renderer=None) -> Size:
        # Build components first, THEN measure and layout
        self.build_diagram()

        # Measure/Layout layers and nodes (non-overlay)
        for layer in self.children:
            if isinstance(layer, Container) and not layer.is_overlay:
                layer.measure_and_layout(renderer)

        # Now measure the main diagram container based on layer sizes/positions
        # This uses the default Container measure/layout logic for the row of layers
        size = super().measure_and_layout(renderer)

        # Measure overlay connections *after* main layout is determined
        for conn in self.children:
            if isinstance(conn, Connection):
                # Ensure parent link is set if connection was added late
                if conn.parent != self:
                    conn.parent = self
                conn.measure_and_layout(renderer)

        return size


# Helper function draw_network_diagram (no changes needed from previous version)
def draw_network_diagram(
    network,
    figsize=(12, 8),
    dpi=200,
    debug=False,
):
    diagram = NetworkDiagram(network=network, id="network-diagram", debug=debug)

    from .matplotlib_renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

    renderer = MatplotlibRenderer(debug=debug)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")  # turn off axis by default for diagrams

    # This single call triggers the build, measure, layout, render cascade
    renderer.render_component(ax, diagram, adjust_lims=True)

    return fig, ax, diagram
