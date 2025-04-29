# File: example/net_genetic.py
# -*- coding: utf-8 -*-
"""example script for drawing network genetic schematics."""

import matplotlib.pyplot as plt
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import pandas as pd
from jeanplot.debug import set_debug, get_logger
from jeanplot.style import jstyle
import dracon as dr
import random
import json
from jeanplot.network_schematic import NetworkGeneticSchematic
from jeanplot.network_utils import get_tu_informations, get_tu_grid_layout, get_interactions
from jeanplot.matplotlib_renderer import MatplotlibRenderer
from jeanplot.container import Container
from jeanplot.models import LayoutConstraints
from jeanplot.text import Text


# --- configuration ---
# set to true for detailed console output from jeanplot components
DEBUG_MODE = True
# select network by index (e.g., 0, 1, ... or -1 for last)
NETWORK_INDEX = 15  # try index 23 for a more complex example
# connection style: 'orthogonal', 'bezier', 'straight'
CONNECTION_STYLE = "bezier"
# layout orientation: 'row' (TUs arranged horizontally) or 'column' (vertically)
LAYOUT_ORIENTATION = "column"
SHOW_ALL_TUS = False
# ---------------------

logger = get_logger(__name__)
set_debug(DEBUG_MODE)

theme = dr.load(
    "pkg:jeanplot:resources/themes/default",
    enable_interpolation=True,
    raw_dict=True,
    use_cache=False,
)
dr.resolve_all_lazy(theme)

jstyle.clear()
jstyle.update(theme)
logger.info("theme loaded and applied successfully.")


logger.info("loading biocomp library...")
try:
    lib = load_lib()
    engine = md.get_biocompdb_sqlite_engine(cm.config.db.sqlite.path)
    logger.info("library loaded.")
except Exception as e:
    logger.error(f"failed to load biocomp library or database: {e}", exc_info=DEBUG_MODE)
    exit()
# define network set selector
netset = NetworkSet(
    content=[NetworkSelector(experiment_name="2023-11-17_PguConstraints1_BP_DR", recipe_name="")]
)
logger.info("running network selectors...")
with Session(engine) as session:
    netset.run_selectors(session)
    data = netset.get_networks_and_data(session)
    networks, _ = zip(*data)  # datafiles not used here
networks = list(networks)
logger.info(f"found {len(networks)} networks.")
if not networks:
    logger.error("no networks found matching selector.")
    exit()
if NETWORK_INDEX >= len(networks) or NETWORK_INDEX < -len(networks):
    logger.warning(
        f"network index {NETWORK_INDEX} out of bounds (0-{len(networks) - 1}). using index 0."
    )
    NETWORK_INDEX = 0
mnet = networks[NETWORK_INDEX]
logger.info(f"processing network: {mnet.name} (index {NETWORK_INDEX})")
mnet.build(lib)
network = mnet._network
# dump network info for debugging if needed
if DEBUG_MODE:
    logger.debug("--- network recipe ---")
    logger.debug(json.dumps(mnet.recipe.content, indent=2))
    logger.debug("--- network info ---")
    logger.debug(json.dumps(network.generate_network_info(), indent=2))
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
        logger.debug("--- central dogma graph ---")
        logger.debug(network.central_dogma_graph)
        logger.debug("--- compute graph ---")
        logger.debug(network.compute_graph)
tu_infos = get_tu_informations(network)
layout_layers = get_tu_grid_layout(network)  # uses topological sort
interactions = get_interactions(network)
if DEBUG_MODE:
    logger.debug("--- tu infos ---")
    logger.debug(json.dumps({k: v.model_dump() for k, v in tu_infos.items()}, indent=2))
    logger.debug("--- raw layout layers ---")
    logger.debug(json.dumps(layout_layers, indent=2))
    logger.debug("--- interactions ---")
    logger.debug(json.dumps([i.model_dump() for i in interactions], indent=2))
logger.info("information extracted.")
# ------------------------------

# --- draw schematic ---
logger.info(f"drawing schematic (style: {CONNECTION_STYLE}, orientation: {LAYOUT_ORIENTATION})...")

schematic = NetworkGeneticSchematic(
    id="network-schematic",
    network=network,
    show_all_tus=SHOW_ALL_TUS,
    layout_orientation=LAYOUT_ORIENTATION,
)
title = Text(
    text=mnet.name,
    font_size=6,
)
root = Container(children=[title, schematic], layout=LayoutConstraints(direction="column", gap=15))

renderer = MatplotlibRenderer()
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.set_aspect("equal")
ax.axis("off")
renderer.render_component(ax, root, adjust_lims=True)
