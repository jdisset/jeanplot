# File: jeanplot/network_diagram_v2.py
# -*- coding: utf-8 -*-
"""
Network compute diagram for new biocomp GraphState-based networks.
Minimal adaptation of network_diagram.py for the new network system.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import Field, PrivateAttr

from jeanplot.component import AnchorComponent
from jeanplot.container import Container
from jeanplot.models import BoxStyle, LayoutConstraints, Offset
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.svg import LineEndFlat
from jeanplot.network_adapter import get_tu_informations_v2
from jeanplot.debug import debug_print, get_logger
from jeanplot.text import Text

logger = get_logger(__name__)


# === Node Components (same as network_diagram.py) ===

class ComputeNode(Container):
    """base class for nodes in the compute graph diagram."""

    node_type: str = "unknown"
    node_label: Optional[str] = None
    node_id: Optional[int] = None
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="center", justify_content="center")
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        if self.node_label:
            self.add_child(
                Text(
                    text=self.node_label,
                    id=f"lbl_{self.id}" if self.id else None,
                    style_class=["label"],
                    vertical_align="middle",
                    align="center",
                )
            )


class TranscriptionNode(ComputeNode):
    node_type: str = "transcription"
    node_label: Optional[str] = "Tx"


class TranslationNode(ComputeNode):
    node_type: str = "translation"
    node_label: Optional[str] = "Tl"


class ERNNode(ComputeNode):
    node_type: str = "sequestron_ERN"
    _tx_node: TranscriptionNode = PrivateAttr()
    _tl_node: TranslationNode = PrivateAttr()
    _out: AnchorComponent = PrivateAttr()
    _center: AnchorComponent = PrivateAttr()
    _tx_connector: Connection = PrivateAttr()
    _tl_connector: Connection = PrivateAttr()

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        tx_id, tl_id, out_id, center_id = (
            f"{prefix}_{self.id}" if self.id else None for prefix in ["tx", "tl", "out", "center"]
        )
        txconn_id, tlconn_id = (
            f"{prefix}_{self.id}" if self.id else None for prefix in ["txconn", "tlconn"]
        )
        self._tx_node = TranscriptionNode(id=tx_id, is_overlay=True)
        self._tl_node = TranslationNode(id=tl_id, is_overlay=True)
        self._out = AnchorComponent(
            id=out_id, style_class=["ernout"], offset=Offset(reference_relative=(1.0, 0.5))
        )
        self._center = AnchorComponent(
            id=center_id, style_class=["erncenter"], offset=Offset(reference_relative=(0.5, 0.5))
        )
        self._tx_connector = Connection(
            id=txconn_id,
            start_component=self._tx_node,
            end_component=self._out,
            style_class=["txconn"],
            curve_type=SimpleBezierCurve(),
            auto_route=False,
        )
        self._tl_connector = Connection(
            id=tlconn_id,
            start_component=self._tl_node,
            end_component=self._center,
            style_class=["tlconn"],
            curve_type=OrthogonalCurve(corner_radius=50, start_length=5, end_length=5),
            end_cap=LineEndFlat(),
            auto_route=False,
        )
        self.add_children([
            self._tx_node, self._tl_node, self._out, self._center,
            self._tx_connector, self._tl_connector,
        ])


class FluoNode(ComputeNode):
    node_type: str = "output"
    node_label: Optional[str] = "Y"


class InvNode(ComputeNode):
    node_type: str = "inverted"
    node_label: Optional[str] = "Inv"


class TUNode(ComputeNode):
    node_type: str = "source"


class AggregationNode(ComputeNode):
    node_type: str = "aggregation"
    collapsed: bool = False


class DeadEndNode(ComputeNode):
    node_type: str = "deadend"
    node_label: Optional[str] = "X"


class InputNode(ComputeNode):
    node_type: str = "input"
    node_label: Optional[str] = "In"


# === Main Diagram Component ===

class NetworkDiagramV2(Container):
    """
    Network compute diagram for GraphState-based networks.
    Minimal adaptation of NetworkDiagram for the new network system.
    """

    network: Any = Field(description="biocomp Network object (GraphState-based)")
    simplified: bool = Field(default=True, description="hide inverse chains and input nodes")

    # top-level layout defaults matching prototype
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row",
            gap=15,
            justify_content="center",
            align_items="stretch",
        )
    )
    style_class: list[str] = ["NetworkDiagram"]
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0), margin=(0, 0, 0, 0))
    )

    _nodes: Dict[int, ComputeNode] = PrivateAttr(default_factory=dict)
    _connections: List[Connection] = PrivateAttr(default_factory=list)
    _layed_out_node_ids: Set[int] = PrivateAttr(default_factory=set)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if not self.network or self.network.compute_graph is None:
            raise ValueError("network with compute_graph is required")

        self._build_nodes()
        self._build_connections()
        layers = self._create_and_position_layers()
        self.children = self._connections + layers

    def _log_debug(self, message: str, data: Any = None):
        comp_id = self.id or self.__class__.__name__
        debug_print(comp_id, message, data)

    # === Adapters for new GraphState API ===

    def _get_node(self, node_id: int):
        """Get node from compute_graph."""
        return self.network.compute_graph.nodes.get(node_id)

    def _get_node_type(self, node_id: int) -> str:
        """Get node type."""
        node = self._get_node(node_id)
        return node.node_type if node else "unknown"

    def _get_outgoing_edges(self, node_id: int) -> List:
        """Get edges where this node is the source."""
        return list(self.network.compute_graph.get_outgoing_edges(node_id))

    def _get_nodes_by_type(self, node_type: str) -> List[int]:
        """Get all node IDs of a given type."""
        return [
            nid for nid, node in self.network.compute_graph.nodes.items()
            if node.node_type == node_type
        ]

    def _compute_dependency_map(self) -> Dict[int, List[int]]:
        """Build dependency map: node_id -> list of upstream node_ids."""
        dep_map = {}
        for edge in self.network.compute_graph.edges.values():
            target_id = edge.target_id
            source_id = edge.source_id
            if target_id not in dep_map:
                dep_map[target_id] = []
            dep_map[target_id].append(source_id)
        return dep_map

    # === Node creation ===

    def _make_node(self, node, node_id: int) -> Optional[ComputeNode]:
        """Factory function to create node components based on type."""
        node_type = node.node_type
        comp_id_str = f"node_{node_id}"
        kw = {"node_id": node_id, "id": comp_id_str}

        node_class_map = {
            "transcription": TranscriptionNode,
            "translation": TranslationNode,
            "output": FluoNode,
            "sequestron_ERN": ERNNode,
            "deadend": DeadEndNode,
            "source": TUNode,
            "input": InputNode,
            "aggregation": AggregationNode,
        }

        if node_type not in node_class_map:
            if node_type.startswith("inv_"):
                return InvNode(**kw)
            return ComputeNode(node_type=node_type, node_label="?", **kw)

        node_class = node_class_map[node_type]

        if node_class is AggregationNode:
            node_label = None
            style_class = ["aggregation"]

            tu_infos = get_tu_informations_v2(self.network)
            markers = set()

            # find markers from connected source nodes
            for edge in self._get_outgoing_edges(node_id):
                target = self._get_node(edge.target_id)
                if target and target.node_type == "source":
                    tu_id = target.extra.get("name")
                    cotx_group = target.extra.get("cotx_group", "cotx_1")
                    full_tu_id = f"{tu_id}_{cotx_group}" if tu_id else None
                    if full_tu_id and full_tu_id in tu_infos:
                        if tu_infos[full_tu_id].cotx_marker:
                            markers.add(tu_infos[full_tu_id].cotx_marker)

            if len(markers) == 1:
                marker = markers.pop()
                if marker:
                    style_class.append(marker)
                    node_label = marker
                    cotx_name = node.extra.get("name")
                    if cotx_name:
                        node_label = cotx_name

            return AggregationNode(style_class=style_class, node_label=node_label, collapsed=True, **kw)

        elif node_class is TUNode:
            tu_infos = get_tu_informations_v2(self.network)
            tu_id = node.extra.get("name")
            cotx_group = node.extra.get("cotx_group", "cotx_1")
            full_tu_id = f"{tu_id}_{cotx_group}" if tu_id else None

            node_label = None
            style_class = ["source"]
            if full_tu_id and full_tu_id in tu_infos:
                marker = tu_infos[full_tu_id].cotx_marker
                if marker:
                    style_class.append(marker)
                    style_class.append("tu_marker")
                    node_label = marker

            return TUNode(style_class=style_class, node_label=node_label, **kw)

        elif node_class is FluoNode:
            net_info = self.network.generate_network_info()
            markers = set(net_info.get("dependent_outputs", []))
            fnode = FluoNode(**kw)
            if len(markers) == 1:
                marker = markers.pop()
                fnode.style_class.append(marker)
            return fnode

        elif node_class is ERNNode:
            ern_name = node.extra.get("seq_name", "")
            ern_name = ern_name.split("::")[-1] if ern_name else ""
            ern_name = ern_name.split("#")[0] if ern_name else ""
            e = ERNNode(**kw)
            e.add_child(Text(text=ern_name, style_class=["ern_name"]))
            return e

        return node_class(**kw)

    def _get_node_filter(self) -> Set[int]:
        """Determine which nodes to exclude based on simplified mode."""
        excluded = set()
        if self.simplified:
            graph = self.network.compute_graph
            for node in graph.nodes.values():
                # exclude inverse nodes
                if node.is_inverse_of is not None:
                    excluded.add(node.node_id)
                    # also exclude the original node if it's not aggregation/source
                    inverted_id = node.is_inverse_of.node_id
                    inverted_node = graph.nodes.get(inverted_id)
                    if inverted_node and inverted_node.node_type not in ("aggregation", "source"):
                        excluded.add(inverted_id)
                # exclude input nodes
                if node.node_type == "input":
                    excluded.add(node.node_id)
        return excluded

    def _build_nodes(self):
        self._nodes.clear()
        graph = self.network.compute_graph
        excluded = self._get_node_filter()
        for node in graph.nodes.values():
            if node.node_id in excluded:
                continue
            comp = self._make_node(node, node.node_id)
            if comp:
                self._nodes[node.node_id] = comp

    def _connect(self, src_id: int, dst_id: int, slot: int) -> Optional[Connection]:
        if src_id not in self._nodes or dst_id not in self._nodes:
            return None
        src_comp = self._nodes[src_id]
        dst_comp = self._nodes[dst_id]
        start_target = getattr(src_comp, "_out", src_comp) if isinstance(src_comp, ERNNode) else src_comp
        style_classes = [
            "comp-connection", f"src-{src_comp.node_type}",
            f"dst-{dst_comp.node_type}", f"slot-{slot}",
        ]
        conn_id = f"conn_{src_id}_{dst_id}_{slot}"
        return Connection(
            id=conn_id, start_component=start_target, end_component=dst_comp,
            line_width=1, style_class=style_classes,
        )

    def _build_connections(self):
        self._connections = []
        graph = self.network.compute_graph
        for edge in graph.edges.values():
            src_id = edge.source_id
            tgt_id = edge.target_id
            if src_id not in self._nodes or tgt_id not in self._nodes:
                continue
            src_type = self._nodes[src_id].node_type
            tgt_type = self._nodes[tgt_id].node_type
            # Skip connections to/from aggregation nodes (they're hidden)
            if src_type == "aggregation" or tgt_type == "aggregation":
                continue
            # Skip connections to ERN (handled internally by ERNNode)
            if tgt_type == "sequestron_ERN":
                continue
            conn = self._connect(src_id, tgt_id, edge.to_input_slot)
            if conn:
                self._connections.append(conn)

    def _create_and_position_layers(self) -> List[Container]:
        """Build layer structure matching the original prototype logic."""
        self._layed_out_node_ids = set()
        all_layers = []
        graph = self.network.compute_graph
        dependency_map = self._compute_dependency_map()
        ern_ids = self._get_nodes_by_type("sequestron_ERN")

        # === Input layer (source nodes only, one per aggregation group) ===
        # Don't show aggregation containers, just organize sources vertically
        # Hidden sources are attached to visible ones so edges appear to come from them
        agg_ids = self._get_nodes_by_type("aggregation")
        src_ids = self._get_nodes_by_type("source")
        input_layer = Container(id="layer_input", style_class=["input_layer", "layer"])
        processed_sources = set()

        # Add ONE representative source per aggregation group, attach others to it
        for agg_id in sorted(agg_ids):
            if agg_id in self._nodes:
                self._layed_out_node_ids.add(agg_id)  # mark as processed but don't add to layer
                visible_source = None
                hidden_sources = []
                for edge in self._get_outgoing_edges(agg_id):
                    if edge.target_id in self._nodes and isinstance(self._nodes[edge.target_id], TUNode):
                        source_node = self._nodes[edge.target_id]
                        processed_sources.add(edge.target_id)
                        self._layed_out_node_ids.add(edge.target_id)
                        if visible_source is None:
                            visible_source = source_node
                            input_layer.add_child(source_node)
                        else:
                            hidden_sources.append(source_node)
                # Attach hidden sources to the visible one
                for hidden_src in hidden_sources:
                    hidden_src.attached_to = visible_source
                    hidden_src.show = False
                    input_layer.add_child(hidden_src)

        # Add any remaining sources not connected to an aggregation
        for src_id in sorted(src_ids):
            if src_id in self._nodes and src_id not in processed_sources:
                input_layer.add_child(self._nodes[src_id])
                self._layed_out_node_ids.add(src_id)

        if input_layer.children:
            all_layers.append(input_layer)

        # === ERN layers ===
        ern_ids_in_nodes = [eid for eid in ern_ids if eid in self._nodes]
        topo_ern_layers = graph.topological_order(ern_ids_in_nodes)
        for i, ern_layer_ids in enumerate(topo_ern_layers):
            if not ern_layer_ids:
                continue
            layer_container = Container(style_class=["main_layer", f"main_layer_{i}", "layer"])
            layer_title = Text(
                text=f"Layer {i + 1}",
                font_size=5,
                style_class=["layer_title"],
                offset=Offset(reference_relative=(0.5, 1), relative=(-0.6, 1.5)),
                is_overlay=True,
                id=f"title_ern_{i}",
            )
            layer_container.add_child(layer_title)

            for ern_id in sorted(ern_layer_ids):
                if ern_id in self._nodes:
                    ern_node = self._nodes[ern_id]
                    layer_container.add_child(ern_node)
                    self._layed_out_node_ids.add(ern_id)
                    # attach upstream (hidden) nodes
                    upstream_ids = dependency_map.get(ern_id, [])
                    for src_id in upstream_ids:
                        if src_id in self._nodes and src_id not in ern_ids:
                            node_to_attach = self._nodes[src_id]
                            attachment_target = None
                            if isinstance(node_to_attach, TranslationNode):
                                attachment_target = ern_node._tl_node
                            elif isinstance(node_to_attach, TranscriptionNode):
                                attachment_target = ern_node._tx_node
                            if attachment_target:
                                node_to_attach.attached_to = attachment_target
                                node_to_attach.show = False
                                layer_container.add_child(node_to_attach)
                                self._layed_out_node_ids.add(src_id)

            if layer_container.children:
                all_layers.append(layer_container)

        # === Output layer ===
        output_ids = self._get_nodes_by_type("output")
        if output_ids:
            out_id = output_ids[0]
            if out_id in self._nodes:
                output_layer = Container(id="layer_output", style_class=["output_layer", "layer"])
                out_node = self._nodes[out_id]
                output_layer.add_child(out_node)
                self._layed_out_node_ids.add(out_id)
                # attach upstream (visible) nodes
                upstream_ids = dependency_map.get(out_id, [])
                for src_id in upstream_ids:
                    if src_id in self._nodes and src_id not in ern_ids:
                        node_to_attach = self._nodes[src_id]
                        if isinstance(node_to_attach, (TranscriptionNode, TranslationNode)):
                            node_to_attach.attached_to = out_node
                            node_to_attach.attachment_offset = Offset(absolute=(-40, 0))
                            node_to_attach.show = True
                            output_layer.add_child(node_to_attach)
                            self._layed_out_node_ids.add(src_id)
                if output_layer.children:
                    all_layers.append(output_layer)

        # === Auto-layout remaining nodes ===
        remaining_ids = set(self._nodes.keys()) - self._layed_out_node_ids - set(ern_ids) - set(output_ids)
        auto_layers = []
        if remaining_ids:
            auto_topo_layers = graph.topological_order(list(remaining_ids))
            for i, auto_layer_ids in enumerate(auto_topo_layers):
                if not auto_layer_ids:
                    continue
                layer_container = Container(id=f"layer_auto_{i}", style_class=["auto_layer", "layer"])
                for node_id in sorted(auto_layer_ids, reverse=True):
                    if node_id in self._nodes:
                        layer_container.add_child(self._nodes[node_id])
                        self._layed_out_node_ids.add(node_id)
                if layer_container.children:
                    auto_layers.append(layer_container)
            # insert auto layers after input layer
            all_layers = all_layers[:1] + auto_layers + all_layers[1:]

        return all_layers
