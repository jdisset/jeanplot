# File: jeanplot/network_diagram.py
# -*- coding: utf-8 -*-
"""
Components and container for drawing network compute diagrams based on biocomp Network objects.
Replicates the structure and logic of the original prototype script more closely.
"""

from typing import Any
from pydantic import Field, PrivateAttr
import pandas as pd

from jeanplot.core.component import AnchorComponent
from jeanplot.core.container import Container
from jeanplot.core.models import BoxStyle, LayoutConstraints, Offset
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.svg import LineEndFlat
from jeanplot._deprecated.network_utils import get_tu_informations
from jeanplot.core.debug import get_logger
from jeanplot.core.text import Text

logger = get_logger(__name__)


class ComputeNode(Container):
    """base class for nodes in the compute graph diagram."""

    node_type: str = "unknown"
    node_label: str | None = None
    node_id: int | None = None
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
    node_label: str | None = "Tx"


class TranslationNode(ComputeNode):
    node_type: str = "translation"
    node_label: str | None = "Tl"


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
        self.add_children(
            [
                self._tx_node,
                self._tl_node,
                self._out,
                self._center,
                self._tx_connector,
                self._tl_connector,
            ]
        )


class FluoNode(ComputeNode):
    node_type: str = "output"
    node_label: str | None = "Y"


class InvNode(ComputeNode):
    node_type: str = "inverted"
    node_label: str | None = "Inv"


class TUNode(ComputeNode):
    node_type: str = "source"


class AggregationNode(ComputeNode):
    node_type: str = "aggregation"
    collapsed: bool = False


class DeadEndNode(ComputeNode):
    node_type: str = "deadend"
    node_label: str | None = "X"


class InputNode(ComputeNode):
    node_type: str = "input"
    node_label: str | None = "In"


class BiasNode(ComputeNode):
    node_type: str = "bias"
    node_label: str | None = "B"


# --- Main Diagram Component ---


class NetworkDiagram(Container):
    """
    container that generates a network compute diagram, closely following the prototype logic.
    """

    network: Any = Field(description="biocomp network object")
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
    style_class: list[str] = ["main_diagram"]
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0), margin=(0, 0, 0, 0))
    )

    # internal state needed during build process
    _nodes: dict[int, ComputeNode] = PrivateAttr(default_factory=dict)
    _connections: list[Connection] = PrivateAttr(default_factory=list)
    _layed_out_node_ids: set[int] = PrivateAttr(default_factory=set)
    _no_dst_connect: set[int] = PrivateAttr(default_factory=set)

    def model_post_init(self, *args, **kwargs):
        """initializes the diagram by processing the network and building components."""
        super().model_post_init(*args, **kwargs)
        self._log_debug("initializing network diagram (prototype replication)...")
        if not self.network:
            raise ValueError("network object is required for NetworkDiagram.")

        self._build_nodes()

        self._build_connections()

        layers = self._create_and_position_layers()

        self.children = self._connections + layers  # type: ignore[assignment]

        self._log_debug("network diagram initialization complete.")

    def _make_node(self, row: pd.Series, node_id: int) -> ComputeNode | None:
        """factory function to create specific node components based on type."""
        node_type = row["type"]
        comp_id_str = f"node_{node_id}"  # unique id for the component
        kw = {"node_id": node_id, "id": comp_id_str}  # common kwargs

        node_class_map = {
            "transcription": TranscriptionNode,
            "translation": TranslationNode,
            "output": FluoNode,
            "sequestron_ERN": ERNNode,
            "deadend": DeadEndNode,
            "source": TUNode,
            "input": InputNode,
            "bias": BiasNode,
            "aggregation": AggregationNode,
        }
        if node_type in node_class_map:
            node_class = node_class_map[node_type]
            if node_class is AggregationNode:
                node_label = None
                style_class = ["aggregation"]

                # Get the marker for styling
                tu_infos = get_tu_informations(self.network)

                # For aggregation nodes, we need to look at the TUs from the connected source nodes
                markers = set()
                output_to = row.get("output_to", [])
                for target_id, _ in output_to:
                    if target_id in self.network.compute_graph.index:
                        target_row = self.network.compute_graph.loc[target_id]
                        if target_row["type"] == "source":
                            # Get TUs from this source node
                            source_cdg_outputs = target_row.get("cdg_output", [])
                            for cdg_idx in source_cdg_outputs:
                                # Look up the TU ID from the central dogma graph
                                if cdg_idx in self.network.central_dogma_graph.index:
                                    tu_id = self.network.central_dogma_graph.loc[cdg_idx]["tu_id"]
                                    if isinstance(tu_id, list) and tu_id:
                                        tu_id = tu_id[0]
                                    if tu_id in tu_infos and tu_infos[tu_id].cotx_marker:
                                        markers.add(tu_infos[tu_id].cotx_marker)
                if len(markers) == 1:
                    marker = markers.pop()
                    if marker:
                        style_class.append(marker)
                        node_label = marker  # Default to marker name

                        # Check if there's a cotransfection name to use instead
                        extra_data = row.get("extra", {})
                        cotx_name = (
                            extra_data.get("name", None) if isinstance(extra_data, dict) else None
                        )
                        if cotx_name:
                            # Replace the marker name with the cotx name
                            node_label = cotx_name

                # Check if connected to bias node (instead of input)
                input_from = row.get("input_from", [])
                for source_id, _ in input_from:
                    if source_id in self.network.compute_graph.index:
                        source_row = self.network.compute_graph.loc[source_id]
                        if source_row["type"] == "bias":
                            style_class.append("bias_connected")
                            break

                return AggregationNode(
                    style_class=style_class, node_label=node_label, collapsed=True, **kw
                )
            elif node_class is TUNode:
                tu_infos = get_tu_informations(self.network)
                marker, cotx_name = None, None
                style_class = ["source"]
                for co in row.get("cdg_output", []):
                    tu_id = self.network.central_dogma_graph.iloc[co]["tu_id"][0]
                    if tu_id in tu_infos:
                        info = tu_infos[tu_id]
                        marker = info.cotx_marker
                        cotx_name = info.cotx_name
                        if marker:
                            style_class.extend([marker, "tu_marker"])
                        break
                tu = TUNode(style_class=style_class, **kw)  # no node_label
                if cotx_name and marker:
                    lbl = Container(
                        style_class=["tu_label_container"],
                        is_overlay=True,
                        layout=LayoutConstraints(direction="row", gap=0),
                    )
                    lbl.add_child(Text(text=f"{cotx_name} (", style_class=["tu_label_prefix"]))
                    lbl.add_child(Text(text=marker, style_class=["tu_label_marker", marker]))
                    lbl.add_child(Text(text=")", style_class=["tu_label_suffix"]))
                    tu.add_child(lbl)
                elif cotx_name:
                    tu.add_child(
                        Text(text=cotx_name, style_class=["tu_label_prefix"], is_overlay=True)
                    )
                elif marker:
                    tu.add_child(
                        Text(text=marker, style_class=["tu_label_marker", marker], is_overlay=True)
                    )
                return tu

            elif node_class is FluoNode:
                network_info = self.network.generate_network_info()
                markers = set(network_info.get("dependent_outputs", []))
                fnode = FluoNode(**kw)
                if len(markers) == 1:
                    marker = markers.pop()
                    fnode.style_class.append(marker)
                return fnode

            elif node_class is ERNNode:
                # add ern name
                ern_name = row.get("extra", {}).get("seq_name")
                ern_name = ern_name.split("::")[-1] if ern_name else ""
                ern_name = ern_name.split("#")[0] if ern_name else ""
                e = ERNNode(**kw)
                e.add_child(
                    Text(
                        text=ern_name,
                        style_class=["ern_name"],
                    )
                )
                return e

            else:
                return node_class(**kw)
        elif node_type.startswith("inv_"):
            return InvNode(**kw)
        else:
            self._log_debug(
                f"warning: unknown node type '{node_type}' for node {node_id}, creating generic node."
            )
            return ComputeNode(node_type=node_type, node_label="?", **kw)

    def _get_node_filter(self) -> set[int]:
        """determines which node ids to exclude based on simplified mode."""
        excluded_node_ids = set()
        if self.simplified:
            cg = self.network.compute_graph
            inverse_nodes = cg[cg["is_inverse_of"] >= 0]
            inverse_node_ids = set(inverse_nodes.index)
            excluded_node_ids.update(inverse_node_ids)
            for nid, row in inverse_nodes.iterrows():
                inverted_id = row["is_inverse_of"]
                inverted_row = cg.loc[inverted_id]
                if inverted_row["type"] not in ("aggregation", "source"):
                    excluded_node_ids.add(inverted_id)
            input_node_ids = set(cg[cg["type"] == "input"].index)
            excluded_node_ids.update(input_node_ids)
            bias_node_ids = set(cg[cg["type"] == "bias"].index)
            excluded_node_ids.update(bias_node_ids)
        return excluded_node_ids

    def _build_nodes(self):
        """creates all compute node components, respecting filters."""
        self._nodes.clear()
        cg = self.network.compute_graph
        excluded_ids = self._get_node_filter()
        for nid, row in cg.iterrows():
            if nid in excluded_ids:
                continue
            node_comp = self._make_node(row, nid)
            if node_comp:
                self._nodes[nid] = node_comp
        self._log_debug(f"built {len(self._nodes)} nodes after filtering.")

    def _connect(self, src_id: int, dst_id: int, slot: int) -> Connection | None:
        """helper function to create a connection component between two nodes."""
        if src_id not in self._nodes or dst_id not in self._nodes:
            return None
        src_comp = self._nodes[src_id]
        dst_comp = self._nodes[dst_id]
        start_target = (
            getattr(src_comp, "_out", src_comp) if isinstance(src_comp, ERNNode) else src_comp
        )
        style_classes = [
            "comp-connection",
            f"src-{src_comp.node_type}",
            f"dst-{dst_comp.node_type}",
            f"slot-{slot}",
        ]
        conn_id = f"conn_{src_id}_{dst_id}_{slot}"
        self._log_debug(f"creating connection {conn_id}: {src_id} -> {dst_id} (slot {slot})")
        return Connection(
            id=conn_id,
            start_component=start_target,
            end_component=dst_comp,
            line_width=1,
            style_class=style_classes,
        )

    def _build_connections(self):
        """creates all connection components between existing nodes."""
        self._connections = []
        cg = self.network.compute_graph
        for src_node_id, row in cg.iterrows():
            if src_node_id not in self._nodes:
                continue
            for tgt_node_id, input_slot in row.get("output_to", []):
                if tgt_node_id not in self._nodes:
                    continue
                tgt_type = self._nodes[tgt_node_id].node_type
                if tgt_type == "sequestron_ERN":
                    continue
                conn = self._connect(src_node_id, tgt_node_id, input_slot)
                if conn:
                    self._connections.append(conn)
        self._log_debug(f"built {len(self._connections)} connections.")

    def _create_and_position_layers(self) -> list[Container]:
        """
        builds the layer structure and positions nodes within them, matching the prototype's logic.
        returns the list of layer containers in the correct order.
        """
        self._layed_out_node_ids = set()
        all_layers_ordered: list[Container] = []
        cg = self.network.compute_graph
        dependency_map = self.network.compute_dependency_map()
        ern_indices = list(cg[cg["type"] == "sequestron_ERN"].index)

        # --- input layer (aggregation nodes + nested source nodes) ---
        agg_node_ids = list(cg[cg["type"] == "aggregation"].index)
        src_node_ids = list(cg[cg["type"] == "source"].index)
        input_layer = Container(
            id="layer_input",
            style_class=["input_layer", "layer"],
        )
        in_agg = set()
        for agg_id in sorted(agg_node_ids):  # sort for consistent order
            if agg_id in self._nodes:
                agg_node = self._nodes[agg_id]
                input_layer.add_child(agg_node)
                self._layed_out_node_ids.add(agg_id)
                # add source nodes as children to the aggregation node
                sources = cg.loc[agg_id].get("output_to", [])
                source_nodes_to_add = []
                for src_id, _ in sources:
                    if src_id in self._nodes and isinstance(self._nodes[src_id], TUNode):
                        source_nodes_to_add.append(self._nodes[src_id])
                        self._layed_out_node_ids.add(src_id)
                        in_agg.add(src_id)
                agg_node.add_children(source_nodes_to_add)

        for src_id in sorted(src_node_ids):  # sort for consistent order
            if src_id in self._nodes and src_id not in in_agg:
                src_node = self._nodes[src_id]
                input_layer.add_child(src_node)
                self._layed_out_node_ids.add(src_id)

        if input_layer.children:
            all_layers_ordered.append(input_layer)

        # --- ern layers ---
        topo_ern_layers_ids = self.network.topological_order(ern_indices)
        for i, ern_layer_ids in enumerate(topo_ern_layers_ids):
            if not ern_layer_ids:
                continue
            layer_nodes_ordered = sorted(list(ern_layer_ids))
            layer_container = Container(
                style_class=["main_layer", f"main_layer_{i}", "layer"],
            )
            layer_title = Text(
                text=f"Layer {i + 1}",
                font_size=5,
                style_class=["layer_title"],
                offset=Offset(reference_relative=(0.5, 1), relative=(-0.6, 1.5)),
                is_overlay=True,
                id=f"title_ern_{i}",
            )
            layer_container.add_child(layer_title)

            for ern_id in layer_nodes_ordered:
                if ern_id in self._nodes:
                    ern_node = self._nodes[ern_id]
                    layer_container.add_child(ern_node)
                    self._layed_out_node_ids.add(ern_id)
                    # attach upstream (hidden) nodes
                    upstream_ids = dependency_map.get(ern_id, [])
                    for src_id in upstream_ids:
                        if src_id in self._nodes and src_id not in ern_indices:
                            node_to_attach = self._nodes[src_id]
                            attachment_target = None
                            if isinstance(node_to_attach, TranslationNode) and hasattr(
                                ern_node, "_tl_node"
                            ):
                                attachment_target = ern_node._tl_node
                            elif isinstance(node_to_attach, TranscriptionNode) and hasattr(
                                ern_node, "_tx_node"
                            ):
                                attachment_target = ern_node._tx_node

                            if attachment_target:
                                node_to_attach.attached_to = attachment_target
                                node_to_attach.show = False  # hide
                                layer_container.add_child(node_to_attach)
                                self._layed_out_node_ids.add(src_id)
            if layer_container.children:
                all_layers_ordered.append(layer_container)

        # --- output layer ---
        output_node_ids = list(cg[cg["type"] == "output"].index)
        if output_node_ids:
            out_id = output_node_ids[0]
            if out_id in self._nodes:
                output_layer = Container(
                    id="layer_output",
                    style_class=["output_layer", "layer"],
                )
                out_node = self._nodes[out_id]
                output_layer.add_child(out_node)
                self._layed_out_node_ids.add(out_id)
                # attach upstream (visible) nodes
                upstream_ids = dependency_map.get(out_id, [])
                upstream_nodes_to_add = []
                for src_id in upstream_ids:
                    if src_id in self._nodes and src_id not in ern_indices:
                        node_to_attach = self._nodes[src_id]
                        if isinstance(node_to_attach, (TranscriptionNode, TranslationNode)):
                            node_to_attach.attached_to = out_node
                            node_to_attach.attachment_offset = Offset(absolute=(-40, 0))
                            node_to_attach.show = True  # make sure it's visible
                            upstream_nodes_to_add.append(node_to_attach)
                            self._layed_out_node_ids.add(src_id)
                output_layer.add_children(upstream_nodes_to_add)  # add attached nodes to the layer
                if output_layer.children:
                    all_layers_ordered.append(output_layer)
            else:
                self._log_debug(f"warning: output node {out_id} not found in built nodes.")

        # --- layer 2: auto-layout layer(s) of remaining nodes ---
        remaining_ids = (
            set(self._nodes.keys())
            - self._layed_out_node_ids
            - set(ern_indices)
            - set(cg[cg["type"] == "output"].index)
        )
        auto_layers = []
        if remaining_ids:
            auto_topo_layers_ids = self.network.topological_order(list(remaining_ids))
            for i, auto_layer_ids in enumerate(auto_topo_layers_ids):
                if not auto_layer_ids:
                    continue
                layer_nodes_ordered = sorted(list(auto_layer_ids))
                layer_nodes_ordered.reverse()
                layer_container = Container(
                    id=f"layer_auto_{i}",
                    style_class=["auto_layer", "layer"],
                )
                for node_id in layer_nodes_ordered:
                    if node_id in self._nodes:
                        layer_container.add_child(self._nodes[node_id])
                        self._layed_out_node_ids.add(node_id)
                if layer_container.children:
                    auto_layers.append(layer_container)

            all_layers_ordered = all_layers_ordered[:1] + auto_layers + all_layers_ordered[1:]

        return all_layers_ordered
