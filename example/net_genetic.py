# net_genetic.py
import matplotlib.pyplot as plt
from biocomp.utils import load_lib
from sqlmodel import Session
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector
import pandas as pd
from jeanplot.debug import set_debug
from jeanplot.style import jstyle
import dracon as dr
import random
import json

from jeanplot.network_schematic import draw_network_schematic, NetworkGeneticSchematic
from jeanplot.network_utils import get_tu_informations, get_tu_grid_layout, get_interactions

# load theme
theme = dr.load("pkg:jeanplot:resources/themes/default", enable_interpolation=True, raw_dict=True)
dr.resolve_all_lazy(theme)
# use update instead of direct assignment
jstyle.update(theme)

# only enable for debugging
set_debug(False)


lib = load_lib()
engine = md.get_biocompdb_sqlite_engine(cm.config.db.sqlite.path)

netset = NetworkSet(
    content=[NetworkSelector(experiment_name="2023-11-17_PguConstraints1_BP_DR", recipe_name="")]
)

with Session(engine) as session:
    netset.run_selectors(session)
    data = netset.get_networks_and_data(session)
    networks, datafiles = zip(*data)

# random shuffle of networks
networks = list(networks)
# random.shuffle(networks)

for mnet in networks[-1:]:
    mnet.recipe.content
    recipe_str = json.dumps(mnet.recipe.content, indent=2)
    mnet.build(lib)
    network = mnet._network
    print("----------------------------------")
    print(f"network: {mnet.name}")
    print(
        f"network info (obtained from network.generate_network_info()): {json.dumps(network.generate_network_info(), indent=2)}"
    )

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
        print("network.central_dogma_graph:")
        print(network.central_dogma_graph)
        print()
        print("network.compute_graph:")
        print(network.compute_graph)

    tu_infos = get_tu_informations(network)
    json_ready = {k: v.model_dump() for k, v in tu_infos.items()}
    print("tu_infos (obtained from calling network_utils.get_tu_informations):")
    print(json.dumps(json_ready, indent=2))
    print("----------------------------------")

    layout = get_tu_grid_layout(network)
    print("layout:")
    print(json.dumps(layout, indent=2))

    interactions = get_interactions(network)
    print("interactions:")
    print(json.dumps([i.model_dump() for i in interactions], indent=2))

    fig, ax, schematic = draw_network_schematic(
        network,
        connection_style="bezier",
        dpi=300,
        # show_all_tus=True,
        layout_orientation="column",
    )

    # ax.text(
    #     0.1,
    #     -0.01,
    #     recipe_str,
    #     horizontalalignment="left",
    #     verticalalignment="top",
    #     transform=ax.transAxes,
    #     fontsize=5,
    #

    # remove the axis
    ax.axis("off")
    ax.set_title(mnet.name)
    # add recipe content as note

    fig.tight_layout()
    plt.show()

    fig.savefig(f"network_schematic_{mnet.name}.png", dpi=300)
