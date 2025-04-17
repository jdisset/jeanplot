## {{{                          --     imports     --
from typing import Dict, List, Optional, Any, Tuple, Literal, Annotated
from pydantic import Field, PrivateAttr, model_validator, BeforeValidator, BaseModel
import logging
import numpy as np
import pandas as pd
from collections import defaultdict


import matplotlib.pyplot as plt
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import dracon as dr
import random
import json

from jeanplot import load_default_theme

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
    NetworkDiagram,
)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     load network     --
DEBUG_MODE = False
NETWORK_INDEX = 2

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

net.compute_graph

##────────────────────────────────────────────────────────────────────────────}}}

load_default_theme()

diagram = NetworkDiagram(network=net)

fig, ax = plt.subplots(figsize=(10, 20), dpi=300)
ax.set_aspect("equal")
ax.axis("off")
renderer = MatplotlibRenderer()
renderer.render_component(ax, diagram)
