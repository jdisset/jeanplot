## {{{                          --     imports     --
from typing import Dict, List, Optional, Any, Tuple, Literal, Annotated
from pydantic import Field, PrivateAttr, model_validator, BeforeValidator, BaseModel
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from jeanplot.component import Component, AnchorComponent
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset, Transform
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.svg import LineEndFlat, LineEndCircle, LineEndArrow
from jeanplot.network_utils import (
    get_tu_informations,
    get_tu_grid_layout,
    get_interactions,
    optimize_grid_for_source_adjacency,
    _get_source_id,
    TUInfo,
    Interaction,
)
from jeanplot.style import jstyle
from jeanplot.debug import debug_print, get_logger, set_debug
from jeanplot.renderer import BaseRenderer
import matplotlib.pyplot as plt
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import dracon as dr
import random
import json
from jeanplot.network_schematic import NetworkGeneticSchematic
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.text import Text
from jeanplot.network_diagram import (
    ComputeNode,
    TranscriptionNode,
    TranslationNode,
    AggregationNode,
    ERNNode,
    InvNode,
    FluoNode,
    DeadEndNode,
    TUNode,
)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     load network     --
DEBUG_MODE = False
NETWORK_INDEX = 29

lib = load_lib()
engine = md.get_biocompdb_sqlite_engine(cm.config.db.sqlite.path)
# define network set selector
netset = NetworkSet(
    content=[NetworkSelector(experiment_name="2023-11-17_PguConstraints1_BP_DR", recipe_name="")]
)
with Session(engine) as session:
    netset.run_selectors(session)
    data = netset.get_networks_and_data(session)
    networks, _ = zip(*data)  # datafiles not used here
networks = list(networks)
if not networks:
    exit()
mnet = networks[NETWORK_INDEX]
mnet.build(lib)
net = mnet._network
# dump network info for debugging if needed

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.max_colwidth",
    None,
):
    print(net.central_dogma_graph)
    print(net.compute_graph)

tu_infos = get_tu_informations(net)
layout_layers = get_tu_grid_layout(net)  # uses topological sort
interactions = get_interactions(net)
netinfo = net.generate_network_info()
if DEBUG_MODE:
    print("--- tu infos ---")
    print(json.dumps({k: v.model_dump() for k, v in tu_infos.items()}, indent=2))
    print("--- raw layout layers ---")
    print(json.dumps(layout_layers, indent=2))
    print("--- interactions ---")
    print(json.dumps([i.model_dump() for i in interactions], indent=2))


ntypes = net.compute_graph["type"].unique()
node_counts = net.compute_graph["type"].value_counts()
ern_nodes = net.compute_graph[net.compute_graph["type"] == "sequestron_ERN"]
ern_indices = ern_nodes.index
topo = net.topological_order(ern_indices)
nlayers = len(topo)
nerns = len(ern_indices)
ern_names = ", ".join(netinfo["ern_names"])

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                  --     show genetic schematic     --
schematic = NetworkGeneticSchematic(
    network=net,
)
info = Container(
    style=BoxStyle(margin=(0, 0, 10, 3)),
    children=[
        Text(
            text=mnet.name,
            font_size=6,
            font_weight="bold",
            style=BoxStyle(margin=(5, 0, 0, 0)),
        ),
        Text(
            text=f"Detected architecture: {netinfo['architecture'].capitalize()}",
            font_size=6,
        ),
        Text(
            text=f"{nlayers} layers, {nerns} ERNs ({ern_names})",
            font_size=6,
        ),
    ],
    layout=LayoutConstraints(direction="column", gap=4),
)

info.children.reverse()


# only implementing "simplified" mode for now, i.e. no inverse chains, always collapsed aggregations.

theme = dr.load(
    "pkg:jeanplot:resources/themes/default",
    enable_interpolation=True,
    raw_dict=True,
    use_cache=False,
)
dr.resolve_all_lazy(theme)
jstyle.clear()
jstyle.update(theme)

net.compute_graph

dependency_map = net.compute_dependency_map()


# add colors to aggregation nodes
aggrows = net.compute_graph[net.compute_graph["type"] == "aggregation"]
for id, cdgout in aggrows.cdg_output.items():
    markers = set([tu_infos[co].cotx_marker for co in cdgout])
    print(f"{id} -> {markers}")

net.get_output_proteins()


def make_node(row, node_id, **kw):
    node_type = row["type"]
    if node_type == "transcription":
        return TranscriptionNode(node_id=node_id, **kw)
    elif node_type == "translation":
        return TranslationNode(node_id=node_id, **kw)
    elif node_type == "aggregation":
        markers = set([tu_infos[co].cotx_marker for co in row["cdg_output"]])
        node_label = None
        style_class = ["aggregation"]
        if len(markers) == 1:
            marker = markers.pop()
            style_class += [f"{marker}"]
            node_label = marker
            print(f"aggregation {node_id} -> {marker}")
        ag = AggregationNode(
            node_id=node_id,
            collapsed=True,
            style_class=style_class,
            node_label=node_label,
            **kw,
        )
        return ag
    elif node_type == "sequestron_ERN":
        return ERNNode(node_id=node_id, **kw)
    elif node_type.startswith("inv_"):
        return InvNode(node_id=node_id, **kw)
    elif node_type == "output":
        markers = set(netinfo["dependent_outputs"])
        fnode = FluoNode(node_id=node_id, **kw)
        if len(markers) == 1:
            fnode.style_class += [f"{markers.pop()}"]
        return fnode
    elif node_type == "deadend":
        return DeadEndNode(node_id=node_id, **kw)
    elif node_type == "source":
        return TUNode(node_id=node_id, **kw)
    elif node_type == "input":
        return None  # in our simplified version, we don't show inputs (instead aggregations are used as the input)
    else:
        raise ValueError(f"Unknown node type: {node_type}")


# in simplified mode, we don't show inverse nodes
# we also don't show the entire inverted path
# (i.e. the inverted chain of input-inv_transcription-inv_translation-inv_agg-inv_source)
# bt we still need to keep the agg and sources (not the inverse ones themselves, just the original ones)
inverse_nodes = net.compute_graph[net.compute_graph["is_inverse_of"] >= 0]
inverse_nodes_ids = inverse_nodes.index
inverted_nodes_ids = []
for nid, row in inverse_nodes.iterrows():
    inverted = row["is_inverse_of"]
    inverted_row = net.compute_graph.loc[inverted]
    if inverted_row["type"] != "aggregation" and inverted_row["type"] != "source":
        inverted_nodes_ids.append(inverted)
        print(f"adding inverted node {nid} ({row['type']})")
    else:
        print(f"skipping inverted node {nid} ({row['type']}) because it's an aggregation")


nodes = {}
for nid, row in net.compute_graph.iterrows():
    if nid in inverted_nodes_ids or nid in inverse_nodes_ids:
        continue
    node = make_node(row, nid, id=f"{nid}")
    if node is None:
        continue
    nodes[nid] = node


layed_out = set()


edges = [
    (src_node, tgt_node, input_slot)
    for src_node, row in net.compute_graph.iterrows()
    for tgt_node, input_slot in row["output_to"]
    if src_node in nodes.keys() and tgt_node in nodes.keys()
]


def connect(src_id, dst_id, slot=0, **kw):
    src = nodes[src_id]
    if isinstance(src, ERNNode):
        src = src._out
    dst = nodes[dst_id]
    style_class = ["comp-connection"]
    style_class += [f"src-{nodes[src_id].node_type}"]
    style_class += [f"dst-{dst.node_type}"]
    style_class += [f"slot-{slot}"]
    print(f"connecting {src_id}, style_class={style_class}")
    return Connection(
        start_component=src, end_component=dst, line_width=1, style_class=style_class, **kw
    )


connections = [connect(src_id, dst_id, slot) for src_id, dst_id, slot in edges]


layers = []
for i, l in enumerate(topo):
    layer = Container(
        layout=LayoutConstraints(
            direction="column",
            gap=15,
            justify_content="space-evenly",
            align_items="center",
        ),
        style_class=[f"main_layer_{i}"],
    )
    layer_title = Text(
        text=f"Layer {len(layers)+1}",
        font_size=7,
        style_class=["layer_title"],
        offset=Offset(reference_relative=(0.5, 1), relative=(-0.6, 1.5)),
        is_overlay=True,
    )
    for ern_id in l:
        ern_node = nodes[ern_id]
        layer.add_child(ern_node)
        layed_out.add(ern_id)
        upstream = dependency_map[ern_id]
        for src_id in upstream:
            if src_id in nodes and src_id not in ern_indices:
                n = nodes[src_id]
                if isinstance(n, TranslationNode):
                    n.attached_to = ern_node._tl_node  # type: ignore
                else:
                    assert isinstance(n, TranscriptionNode)
                    n.attached_to = ern_node._tx_node  # type: ignore
                n.show = False
                layer.add_child(n)
                layed_out.add(src_id)
    layer.add_child(layer_title)
    layers.append(layer)


# last layer is output
out_id = net.compute_graph.loc[net.compute_graph["type"] == "output"].index[0]
out_node = nodes[out_id]
layers.append(
    Container(
        children=[out_node],
        layout=LayoutConstraints(
            direction="column",
            gap=15,
            justify_content="center",
            align_items="center",
        ),
        style=BoxStyle(
            margin=(0, 0, 0, 30),
            padding=(0, 0, 0, 0),
            border_width=5,
        ),
        style_class=["output_layer"],
    )
)
layed_out.add(out_id)
upstream = dependency_map[out_id]
for src_id in upstream:
    if src_id in nodes and src_id not in ern_indices:
        n = nodes[src_id]
        n.attached_to = out_node
        n.attachment_offset = Offset(absolute=(-40, 0))

        layers[-1].add_child(n)
        layed_out.add(src_id)


# put sources inside aggregation nodes
agg_nodes = net.compute_graph[net.compute_graph["type"] == "aggregation"]
for agg_id, row in agg_nodes.iterrows():
    agg_node = nodes[agg_id]
    sources = row["output_to"]
    print(sources)
    for src_id, slot in sources:
        agg_node.add_child(nodes[src_id])
        layed_out.add(src_id)


# input layer, i.e. aggregation nodes in the simplified version
input_layer = Container(
    layout=LayoutConstraints(
        direction="column",
        gap=15,
        justify_content="space-evenly",
        align_items="center",
    ),
    style_class=["input_layer"],
)
for agg_id, row in agg_nodes.iterrows():
    agg_node = nodes[agg_id]
    input_layer.add_child(agg_node)
    layed_out.add(agg_id)


# remaining nodes
net.compute_graph
remaining = set(nodes.keys()) - set(layed_out)
prev_layers = net.topological_order(remaining)
auto_layers = []
for i, l in enumerate(prev_layers):
    layer_children = [nodes[i] for i in l]
    layer_children.reverse()
    new_layer = Container(
        layout=LayoutConstraints(
            direction="column", gap=15, justify_content="space-between", align_items="start"
        ),
        style=BoxStyle(
            margin=(0, 0, 0, 0),
            padding=(0, 0, 0, 0),
        ),
        style_class=[f"auto_layer_{i}"],
        children=layer_children,
    )
    layed_out.update(l)
    auto_layers.append(new_layer)


children = connections + [input_layer] + auto_layers + layers

diagram_root = Container(
    children=children,
    layout=LayoutConstraints(
        direction="row",
        gap=15,
        justify_content="center",
        align_items="stretch",
    ),
    style_class=["main_diagram"],
    style=BoxStyle(
        margin=(0, 0, 0, 0),
        padding=(0, 0, 0, 0),
    ),
)

schema_root = Container(
    children=[info, schematic], layout=LayoutConstraints(direction="column", gap=15)
)

diagram_root.measure_and_layout()

root = Container(
    children=[schema_root, diagram_root],
    layout=LayoutConstraints(
        direction="column",
        gap=30,
        justify_content="center",
        align_items="stretch",
    ),
)

mnet.recipe_name
mnet.recipe.name
mnet.recipe.content.get("name")
mnet.recipe.experiment.errors
mnet.recipe.experiment.content.get("tx_operator")
mnet.recipe.experiment.content.get("machine")
mnet.recipe.experiment.content.get("cell_line")
mnet.recipe.experiment.content.get("transfection_protocol")

fig, ax = plt.subplots(figsize=(10, 20), dpi=300)
ax.set_aspect("equal")
ax.axis("off")
renderer = MatplotlibRenderer()
renderer.render_component(ax, root, adjust_lims=True)
