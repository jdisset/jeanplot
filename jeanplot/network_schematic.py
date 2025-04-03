"""
Some context:
There can be multiple Transcription Units (TU) in a source. Often, in synthetic biology, you add a transfection unit that codes simply for a fluorescent gene, and "tie" it in a cotransfection (cotx) in a known ratio, or on a plasmid at a certain position, to the main "useful" transcription units.
Btw cotx can also contain L2s (plasmids with multiple TUs) and the same logic applies.
I could represent all sources with all the transcription units inside of them. That's a valid and exhaustive way to represent things (the flag "show_all_tus"  will allow to represent every TUs of a source when set to true).
However, by convention, to not overload the schematics, in a source that has a fluo marker in it (as is usually the case) we tend to only plot the non-marker transcription units.
I then add a source_tag (and color the border of the source thanks to the style sheet) that just "discretely" tell what color is attached to this construct and in what ratio (or at what position if it's an L2).
Recognizing a marker transcription unit is easy: it's the simplest possible TU: promoter + fluo gene + terminator (+ maybe insulator I guess).
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Literal
from pydantic import BaseModel, Field, PrivateAttr
from collections import defaultdict

from jeanplot.component import Component
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.genetic_elements import (
    ERN,
    Promoter,
    TranscriptionUnit,
    Terminator,
    UorfGroup,
    Source,
    ERN5pRecog,
    GeneticPart,
    FluoMarker,
)
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.svg import LineEndFlat, LineEndArrow

from .network_utils import get_tu_informations, get_tu_grid_layout, get_interactions
from .network_utils import TUInfo, Interaction, PartInfo
from .text import Text


class NetworkGeneticSchematic(Container):
    """Container for rendering a genetic network schematic"""

    network: Any
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(direction="column", gap=40, align_items="center")
    )
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(background_color="#ffffff", padding=(30, 30, 30, 30))
    )

    show_all_tus: bool = False  # show all TUs in sources
    layout_direction: Literal["row", "column"] = "column"
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    show_legend: bool = False

    _tu_infos: Dict[str, TUInfo] = {}
    _grid_layout: List[List[str]] = []
    _interactions: List[Interaction] = []
    _source_to_tus: Dict[str, List[str]] = PrivateAttr(default_factory=lambda: defaultdict(list))
    _part_components: Dict[Tuple[str, str], Component] = {}

    def model_post_init(self, *args, **kwargs):
        """Initialize after model creation"""
        self._tu_infos = get_tu_informations(self.network)
        self._grid_layout = get_tu_grid_layout(self.network)
        self._interactions = get_interactions(self.network)

        self._group_tus_by_source()
        self._create_layout()

    def _group_tus_by_source(self):
        """Group TUs by their source"""
        for tu_id, tu_info in self._tu_infos.items():
            # create a unique source identifier
            if tu_info.in_l2:
                # L2 plasmids are identified by plasmid name
                source_id = f"plasmid_{tu_info.plasmid_name}"
            else:
                # Non-L2 sources identified by (marker, aggregation_node, ratio)
                source_id = f"source_{tu_info.cotx_marker}_{tu_info.aggregation_node_id}_{tu_info.aggregation_ratio_label}"

            self._source_to_tus[source_id].append(tu_id)

    def _create_layout(self):
        """Create layout of sources containing TUs"""
        grouped_containers = []  # will contain each row or column
        source_ids_processed = set()

        for row_idx, row_tus in enumerate(self._grid_layout):
            sources_in_row = []

            # first, identify sources represented in this row
            row_sources = set()
            for tu_id in row_tus:
                if tu_id not in self._tu_infos:
                    continue

                tu_info = self._tu_infos[tu_id]
                if tu_info.in_l2:
                    source_id = f"plasmid_{tu_info.plasmid_name}"
                else:
                    source_id = f"source_{tu_info.cotx_marker}_{tu_info.aggregation_node_id}_{tu_info.aggregation_ratio_label}"

                row_sources.add(source_id)

            # create sources for this row
            for source_id in row_sources:
                if source_id in source_ids_processed:
                    continue  # skip if already processed

                source_ids_processed.add(source_id)
                tu_ids = self._source_to_tus[source_id]

                # filter out marker TUs if needed
                if not self.show_all_tus:
                    tu_ids = [tu_id for tu_id in tu_ids if not self._tu_infos[tu_id].is_marker]

                if not tu_ids:
                    continue  # skip empty sources

                # create source with all TUs
                source = self._create_source(source_id, tu_ids)
                if source and source.children:
                    sources_in_row.append(source)

            # create a container for this row if it has sources
            if sources_in_row:
                direction = self.layout_direction
                row_container = Container(
                    style_class=["source_row"],
                    layout=LayoutConstraints(
                        direction=direction,
                        gap=40,
                        align_items="center",
                        justify_content="space-evenly",
                    ),
                    children=sources_in_row,
                )

                if self.show_legend:
                    row_title = Text(
                        id=f"row_title_{row_idx}",
                        text=f"Group {row_idx + 1}",
                        font_size=12,
                        color="#555555",
                        align="center",
                    )

                    group_container = Container(
                        style_class=["group_container"],
                        layout=LayoutConstraints(direction="column", gap=10, align_items="center"),
                        children=[row_title, row_container],
                    )

                    grouped_containers.append(group_container)
                else:
                    grouped_containers.append(row_container)

        opposite_direction = "column" if self.layout_direction == "row" else "row"
        if grouped_containers:
            if self.show_legend:
                main_title = Text(
                    id="network_title",
                    text="Network Components",
                    font_size=14,
                    font_weight="bold",
                    color="#333333",
                    align="center",
                    style=BoxStyle(margin=(0, 0, 20, 0)),
                )

                main_container = Container(
                    style_class=["network_schematic"],
                    layout=LayoutConstraints(
                        direction=opposite_direction, gap=60, align_items="center"
                    ),
                    children=[main_title] + grouped_containers,
                )

                self.add_child(main_container)
            else:
                main_container = Container(
                    style_class=["network_schematic"],
                    layout=LayoutConstraints(
                        direction=opposite_direction, gap=60, align_items="center"
                    ),
                    children=grouped_containers,
                )

                self.add_child(main_container)

        self._add_connections()

    def _create_source(self, source_id, tu_ids):
        """Create a source containing multiple TUs"""
        if not tu_ids:
            return None

        first_tu = self._tu_infos.get(tu_ids[0])
        if not first_tu:
            return None

        is_plasmid = first_tu.in_l2
        marker = first_tu.cotx_marker

        if is_plasmid:
            tag_label = f"{marker} L2"
        elif first_tu.aggregation_ratio_label:
            tag_label = f"{marker} {first_tu.aggregation_ratio_label}"
        else:
            tag_label = f"{marker}"

        source = Source(
            source_type="plasmid" if is_plasmid else "cotx",
            marker=marker,
            tag_label=tag_label,
        )

        # add each TU
        for tu_id in tu_ids:
            tu_info = self._tu_infos.get(tu_id)
            if not tu_info:
                continue

            tu = self._create_tu_component(tu_info)
            source.add_child(tu)

        return source if source.children else None

    def _create_tu_component(self, tu_info):
        """Create a transcription unit with genetic elements in correct order"""
        tu = TranscriptionUnit(id=tu_info.tu_id)

        sorted_parts = tu_info.parts  # they should already be sorted

        for part in sorted_parts:
            element = self._create_genetic_element(part, tu_info.tu_id)
            if element:  # skip None elements
                tu.add_child(element)
                self._part_components[(tu_info.tu_id, part.name)] = element

        return tu

    def _create_genetic_element(self, part_info, tu_id):
        """Create appropriate genetic element based on part category"""
        category = part_info.category
        name = part_info.name

        if category == "ERN":
            return ERN(part_name=name, id=f"{tu_id}_{name}")
        elif category == "ERN_recog_site_5p":
            return ERN5pRecog(part_name=name, id=f"{tu_id}_{name}")
        elif category == "promoter":
            return Promoter(id=f"{tu_id}_{name}")
        elif category == "terminator":
            return Terminator(id=f"{tu_id}_{name}")
        elif category == "fluo_marker":
            return FluoMarker(part_name=name, id=f"{tu_id}_{name}")
        elif category == "insulator":
            # skip insulators in visualization
            return None
        elif category == "uORF":
            return UorfGroup(id=f"{tu_id}_{name}")
        else:
            # generic part for anything else
            return GeneticPart(part_type=category, part_name=name, id=f"{tu_id}_{name}")

    def _add_connections(self):
        """Add connections for interactions between genetic elements"""
        for idx, interaction in enumerate(self._interactions):
            # get components for source and target parts
            src_part = self._part_components.get((interaction.src_tu_id, interaction.src_part_name))
            tgt_part = self._part_components.get((interaction.tgt_tu_id, interaction.tgt_part_name))

            if not src_part or not tgt_part:
                continue

            if self.connection_style == "orthogonal":
                curve_type = OrthogonalCurve()
            elif self.connection_style == "bezier":
                curve_type = SimpleBezierCurve()
            else:  # straight
                curve_type = StraightCurve()

            end_cap = LineEndFlat(
                length=8,
            )

            connection = Connection(
                id=f"connection_{idx}",
                start_component=src_part,
                end_component=tgt_part,
                curve_type=curve_type,
                end_cap=end_cap,
            )

            self.add_child(connection)


def draw_network_schematic(
    network,
    figsize=(12, 8),
    dpi=200,
    show_all_tus=False,
    layout_direction="column",
    connection_style="orthogonal",
    show_legend=False,
    debug=False,
):
    schematic = NetworkGeneticSchematic(
        network=network,
        show_all_tus=show_all_tus,
        layout_direction=layout_direction,
        connection_style=connection_style,
        show_legend=show_legend,
    )

    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

    renderer = MatplotlibRenderer(debug=debug)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")

    schematic.measure_and_layout(renderer)
    renderer.render_component(ax, schematic, adjust_lims=True)

    # plt.grid(True, alpha=0.5, linestyle=":")
    # plt.tight_layout()

    return fig, ax, schematic
