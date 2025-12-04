"""Network compute diagram for GraphState-based networks"""

from typing import Dict, List, Optional, Any, Set
from pydantic import Field, PrivateAttr
from collections import defaultdict

from jeanplot.component import AnchorComponent
from jeanplot.container import Container
from jeanplot.models import BoxStyle, LayoutConstraints, Offset
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.svg import LineEndFlat
from jeanplot.network_adapter import get_tu_informations_v2
from jeanplot.text import Text


class ComputeNode(Container):
    node_type: str = "unknown"
    node_label: Optional[str] = None
    node_id: Optional[int] = None
    layout: LayoutConstraints = Field(default_factory=lambda: LayoutConstraints(align_items="center", justify_content="center"))

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        if self.node_label:
            self.add_child(Text(
                text=self.node_label, id=f"lbl_{self.id}" if self.id else None,
                style_class=["label"], vertical_align="middle", align="center",
            ))


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
        mk_id = lambda p: f"{p}_{self.id}" if self.id else None
        self._tx_node = TranscriptionNode(id=mk_id("tx"), is_overlay=True)
        self._tl_node = TranslationNode(id=mk_id("tl"), is_overlay=True)
        self._out = AnchorComponent(id=mk_id("out"), style_class=["ernout"], offset=Offset(reference_relative=(1.0, 0.5)))
        self._center = AnchorComponent(id=mk_id("center"), style_class=["erncenter"], offset=Offset(reference_relative=(0.5, 0.5)))
        self._tx_connector = Connection(
            id=mk_id("txconn"), start_component=self._tx_node, end_component=self._out,
            style_class=["txconn"], curve_type=SimpleBezierCurve(), auto_route=False,
        )
        self._tl_connector = Connection(
            id=mk_id("tlconn"), start_component=self._tl_node, end_component=self._center,
            style_class=["tlconn"], curve_type=OrthogonalCurve(corner_radius=50, start_length=5, end_length=5),
            end_cap=LineEndFlat(), auto_route=False,
        )
        self.add_children([self._tx_node, self._tl_node, self._out, self._center, self._tx_connector, self._tl_connector])


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


NODE_CLASSES = {
    "transcription": TranscriptionNode, "translation": TranslationNode,
    "output": FluoNode, "sequestron_ERN": ERNNode, "deadend": DeadEndNode,
    "source": TUNode, "input": InputNode, "aggregation": AggregationNode,
}


class NetworkDiagramV2(Container):
    network: Any = Field(description="biocomp Network object (GraphState-based)")
    simplified: bool = Field(default=True, description="hide inverse chains and input nodes")
    layout: LayoutConstraints = Field(default_factory=lambda: LayoutConstraints(direction="row", gap=15, justify_content="center", align_items="stretch"))
    style_class: list[str] = ["NetworkDiagram"]
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0), margin=(0, 0, 0, 0)))

    _nodes: Dict[int, ComputeNode] = PrivateAttr(default_factory=dict)
    _connections: List[Connection] = PrivateAttr(default_factory=list)
    _layed_out: Set[int] = PrivateAttr(default_factory=set)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if not self.network or self.network.compute_graph is None:
            raise ValueError("network with compute_graph is required")
        self._build_nodes()
        self._build_connections()
        self.children = self._connections + self._create_layers()

    @property
    def _graph(self):
        return self.network.compute_graph

    def _nodes_by_type(self, t: str) -> List[int]:
        return [n.node_id for n in self._graph.nodes.values() if n.node_type == t]

    def _dep_map(self) -> Dict[int, List[int]]:
        dm = defaultdict(list)
        for e in self._graph.edges.values():
            dm[e.target_id].append(e.source_id)
        return dict(dm)

    def _make_node(self, node, nid: int) -> Optional[ComputeNode]:
        kw = {"node_id": nid, "id": f"node_{nid}"}
        ntype = node.node_type

        if ntype not in NODE_CLASSES:
            return InvNode(**kw) if ntype.startswith("inv_") else ComputeNode(node_type=ntype, node_label="?", **kw)

        cls = NODE_CLASSES[ntype]

        if cls is AggregationNode:
            tu_infos = get_tu_informations_v2(self.network)
            markers = set()
            for e in self._graph.get_outgoing_edges(nid):
                tgt = self._graph.nodes.get(e.target_id)
                if tgt and tgt.node_type == "source":
                    tu_id = f"{tgt.extra.get('name')}_{tgt.extra.get('cotx_group', 'cotx_1')}"
                    if tu_id in tu_infos and tu_infos[tu_id].cotx_marker:
                        markers.add(tu_infos[tu_id].cotx_marker)
            style = ["aggregation"]
            label = None
            if len(markers) == 1 and (m := markers.pop()):
                style.append(m)
                label = node.extra.get("name") or m
            return AggregationNode(style_class=style, node_label=label, collapsed=True, **kw)

        if cls is TUNode:
            tu_infos = get_tu_informations_v2(self.network)
            tu_id = f"{node.extra.get('name')}_{node.extra.get('cotx_group', 'cotx_1')}"
            style, label = ["source"], None
            if tu_id in tu_infos and (m := tu_infos[tu_id].cotx_marker):
                style.extend([m, "tu_marker"])
                label = m
            return TUNode(style_class=style, node_label=label, **kw)

        if cls is FluoNode:
            markers = set(self.network.generate_network_info().get("dependent_outputs", []))
            f = FluoNode(**kw)
            if len(markers) == 1:
                f.style_class.append(markers.pop())
            return f

        if cls is ERNNode:
            ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
            e = ERNNode(**kw)
            e.add_child(Text(text=ern_name, style_class=["ern_name"]))
            return e

        return cls(**kw)

    def _get_excluded(self) -> Set[int]:
        if not self.simplified:
            return set()
        excluded = set()
        for n in self._graph.nodes.values():
            if n.is_inverse_of is not None:
                excluded.add(n.node_id)
                inv = self._graph.nodes.get(n.is_inverse_of.node_id)
                if inv and inv.node_type not in ("aggregation", "source"):
                    excluded.add(inv.node_id)
            if n.node_type == "input":
                excluded.add(n.node_id)
        return excluded

    def _build_nodes(self):
        self._nodes.clear()
        excluded = self._get_excluded()
        for n in self._graph.nodes.values():
            if n.node_id not in excluded and (comp := self._make_node(n, n.node_id)):
                self._nodes[n.node_id] = comp

    def _build_connections(self):
        self._connections = []
        for e in self._graph.edges.values():
            src, tgt = e.source_id, e.target_id
            if src not in self._nodes or tgt not in self._nodes:
                continue
            src_t, tgt_t = self._nodes[src].node_type, self._nodes[tgt].node_type
            if src_t == "aggregation" or tgt_t == "aggregation" or tgt_t == "sequestron_ERN":
                continue
            src_comp = self._nodes[src]
            start = src_comp._out if isinstance(src_comp, ERNNode) else src_comp
            self._connections.append(Connection(
                id=f"conn_{src}_{tgt}_{e.to_input_slot}",
                start_component=start, end_component=self._nodes[tgt],
                line_width=1, style_class=["comp-connection", f"src-{src_t}", f"dst-{tgt_t}", f"slot-{e.to_input_slot}"],
            ))

    def _create_layers(self) -> List[Container]:
        self._layed_out = set()
        layers = []
        dep_map = self._dep_map()
        ern_ids = set(self._nodes_by_type("sequestron_ERN"))

        # input layer
        input_layer = Container(id="layer_input", style_class=["input_layer", "layer"])
        processed_src = set()
        for agg_id in sorted(self._nodes_by_type("aggregation")):
            if agg_id in self._nodes:
                self._layed_out.add(agg_id)
                visible = None
                for e in self._graph.get_outgoing_edges(agg_id):
                    if e.target_id in self._nodes and isinstance(self._nodes[e.target_id], TUNode):
                        src = self._nodes[e.target_id]
                        processed_src.add(e.target_id)
                        self._layed_out.add(e.target_id)
                        if visible is None:
                            visible = src
                            input_layer.add_child(src)
                        else:
                            src.attached_to, src.show = visible, False
                            input_layer.add_child(src)

        for sid in sorted(self._nodes_by_type("source")):
            if sid in self._nodes and sid not in processed_src:
                input_layer.add_child(self._nodes[sid])
                self._layed_out.add(sid)

        if input_layer.children:
            layers.append(input_layer)

        # ERN layers
        ern_in_nodes = [e for e in ern_ids if e in self._nodes]
        for i, ern_layer in enumerate(self._graph.topological_order(ern_in_nodes)):
            if not ern_layer:
                continue
            lc = Container(style_class=["main_layer", f"main_layer_{i}", "layer"])
            lc.add_child(Text(
                text=f"Layer {i + 1}", font_size=5, style_class=["layer_title"],
                offset=Offset(reference_relative=(0.5, 1), relative=(-0.6, 1.5)), is_overlay=True, id=f"title_ern_{i}",
            ))
            for eid in sorted(ern_layer):
                if eid not in self._nodes:
                    continue
                ern = self._nodes[eid]
                lc.add_child(ern)
                self._layed_out.add(eid)
                for uid in dep_map.get(eid, []):
                    if uid in self._nodes and uid not in ern_ids:
                        up = self._nodes[uid]
                        attach = ern._tl_node if isinstance(up, TranslationNode) else (ern._tx_node if isinstance(up, TranscriptionNode) else None)
                        if attach:
                            up.attached_to, up.show = attach, False
                            lc.add_child(up)
                            self._layed_out.add(uid)
            if lc.children:
                layers.append(lc)

        # output layer
        out_ids = self._nodes_by_type("output")
        if out_ids and (oid := out_ids[0]) in self._nodes:
            ol = Container(id="layer_output", style_class=["output_layer", "layer"])
            out = self._nodes[oid]
            ol.add_child(out)
            self._layed_out.add(oid)
            for uid in dep_map.get(oid, []):
                if uid in self._nodes and uid not in ern_ids:
                    up = self._nodes[uid]
                    if isinstance(up, (TranscriptionNode, TranslationNode)):
                        up.attached_to, up.attachment_offset, up.show = out, Offset(absolute=(-40, 0)), True
                        ol.add_child(up)
                        self._layed_out.add(uid)
            if ol.children:
                layers.append(ol)

        # auto-layout remaining
        remaining = set(self._nodes) - self._layed_out - ern_ids - set(out_ids)
        if remaining:
            auto = []
            for i, al in enumerate(self._graph.topological_order(list(remaining))):
                if not al:
                    continue
                ac = Container(id=f"layer_auto_{i}", style_class=["auto_layer", "layer"])
                for nid in sorted(al, reverse=True):
                    if nid in self._nodes:
                        ac.add_child(self._nodes[nid])
                        self._layed_out.add(nid)
                if ac.children:
                    auto.append(ac)
            layers = layers[:1] + auto + layers[1:]

        return layers
