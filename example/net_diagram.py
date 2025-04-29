## {{{                          --     imports     --
from typing import Dict, List, Optional, Any, Tuple, Literal, Annotated
import os
from pydantic import Field, PrivateAttr, model_validator, BeforeValidator, BaseModel
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import dracon as dr
import random
import json
from jeanplot.debug import set_debug, get_logger

from jeanplot import load_default_theme
from jeanplot.biocomp_diagrams import (
    render_network_card,
    render_network_diagram,
    render_circuit_schematic,
)

##────────────────────────────────────────────────────────────────────────────}}}

load_default_theme()
SAVE = True
# SAVE = False
DEBUG_MODE = False
NETWORK_INDEX = 50
# XP_NAME = "2023-02-16_Matrix"
XP_NAME = ""
RECIPE_NAME = ""
# BASE_DIR = Path("~/Dropbox (MIT)/Biocomp_v2/Plots/network_cards").expanduser()
BASE_DIR = Path("/tmp/network_cards").expanduser()
OVERWRITE = True
FIGSIZE = 10

logger = get_logger(__name__)
set_debug(DEBUG_MODE)

lib = load_lib()
engine = md.get_biocompdb_sqlite_engine(cm.config.db.sqlite.path)
netset = NetworkSet(
    content=[NetworkSelector(experiment_name=XP_NAME, recipe_name=RECIPE_NAME)],
)
with Session(engine) as session:
    netset.run_selectors(session)
    data = netset.get_networks_and_data(session)
    networks, _ = zip(*data)  # datafiles not used here
networks = list(networks)


def has_oneone_ratio(ratio_str):
    if not ratio_str:
        return True
    # check that all lines end with 1:1
    return all(line.endswith("1:1") for line in ratio_str.split("\n"))


##
load_default_theme()
NETWORK_INDEX = 30
mnet = networks[NETWORK_INDEX]
# --- Extract Metadata ---
infos = {
    "recipe": mnet.recipe.content.get("name", "N/A").rstrip(),
    "experiment": mnet.recipe.experiment.content.get("name", "N/A").rstrip(),
    "operator": mnet.recipe.experiment.content.get("tx_operator", "N/A"),
    "machine": mnet.recipe.experiment.content.get("machine", "N/A"),
    "cell line": mnet.recipe.experiment.content.get("cell_line", "N/A"),
    "protocol": mnet.recipe.experiment.content.get("transfection_protocol", "N/A"),
}
recipe_txt = mnet.recipe.content
network_info = mnet.network.generate_network_info()

print(network_info.get("cotx_str"))
print(f"Network_info: {network_info}")
print(f"Recipe: {recipe_txt}")
print(f"Infos: {infos}")


fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE), dpi=300)

# render_circuit_schematic(mnet, lib, ax)
# render_network_card(mnet, lib, ax, show_recipe=True)
render_network_diagram(mnet, lib, ax)

mnet.network.compute_graph
mnet.network.central_dogma_graph
# Network Diagram: $BIOCOMP_ROOT/Plots/auto/network_diagram/<network_name>.svg
# Circuit Schematic: $BIOCOMP_ROOT/Plots/auto/circuit_schematic/<network_name>.svg
# Data Plot: $BIOCOMP_ROOT/Plots/auto/data-plot/<data_file_name_stem>.svg

##
# get BIOCOMP_ROOT from env
BIOCOMP_ROOT = Path(os.environ["BIOCOMP_ROOT"]).expanduser()


def make_and_save(mnet, lib, func, fname, **kw):
    fname.parent.mkdir(parents=True, exist_ok=True)

    exists = fname.exists()
    if exists and not OVERWRITE and SAVE:
        print(f"Skipping {fname} as it already exists.")
        return
    fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE), dpi=300)
    try:
        func(mnet, lib, ax, **kw)
        if SAVE:
            fig.savefig(
                fname,
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
        else:
            plt.show()
        plt.close(fig)
        plt.close("all")
    except Exception as e:
        print(f"Error rendering {mnet.name}: {e}")
        plt.close("all")


for i, mnet in tqdm(list(enumerate(networks[:])), desc="Processing networks"):
    ninfo = mnet.network.generate_network_info()

    fname = BIOCOMP_ROOT / "Plots" / "auto" / "network_diagram" / f"{mnet.name}.svg"
    make_and_save(
        mnet=mnet,
        lib=lib,
        func=render_network_diagram,
        fname=fname,
    )

    # fname = BIOCOMP_ROOT / "Plots" / "auto" / "circuit_schematic" / f"{mnet.name}.svg"
    # make_and_save(
    #     mnet=mnet,
    #     lib=lib,
    #     func=render_circuit_schematic,
    #     fname=fname,
    # )
