"""
Some context:
There can be multiple Transcription Units (TU) in a source. Often, in synthetic biology, you add a transfection unit that codes simply for a fluorescent gene, and "tie" it in a cotransfection (cotx) in a known ratio, or on a plasmid at a certain position, to the main "useful" transcription units.
Btw cotx can also contain L2s (plasmids with multiple TUs) and the same logic applies.
I could represent all sources with all the transcription units inside of them. That's a valid and exhaustive way to represent things (the flag "show_all_tus"  will allow to represent every TUs of a source when set to true).
However, by convention, to not overload the schematics, in a source that has a fluo marker in it (as is usually the case) we tend to only plot the non-marker transcription units.
I then add a source_tag (and color the border of the source thanks to the style sheet) that just "discretely" tell what color is attached to this construct and in what ratio (or at what position if it's an L2).
Recognizing a marker transcription unit is easy: it's the simplest possible TU: promoter + fluo gene + terminator (+ maybe insulator I guess).
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from pydantic import Field, PrivateAttr, model_validator
import logging

from jeanplot.component import Component, AnchorComponent
from jeanplot.container import Container
from jeanplot.models import Size, BoxStyle, LayoutConstraints, Offset, Transform
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
    PartInfo,
)
from jeanplot.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.svg import LineEndFlat
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
from jeanplot.debug import debug_print
from jeanplot.renderer import BaseRenderer

logger = logging.getLogger(__name__)


class SourceAnnotation(Container):
    """visual bounding box and tag for a source group in the grid."""

    is_overlay: bool = True
    source_type: Optional[Literal["plasmid", "cotx"]] = "cotx"
    marker: Optional[str] = None
    tag_label: Optional[str] = None
    style_class: list[str] = ["source_annotation"]
    _source_proxy: Optional[Source] = PrivateAttr(default=None)
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._source_proxy = Source(
            source_type=self.source_type,
            marker=self.marker,
            tag_label=self.tag_label,
            children=[],  # proxy doesn't need children
            parent=self,  # set parent link for potential context
        )
        jstyle.apply(self._source_proxy)  # apply styles to proxy to get its styled tag
        tag_cont = next(
            (c for c in self._source_proxy.children if "source_tag" in c.style_class), None
        )
        if tag_cont:
            # important: remove the tag from the proxy before adding to self to avoid double parenting
            self._source_proxy.children.remove(tag_cont)
            tag_cont.parent = self  # ensure parent link is correct
            self.add_child(tag_cont)


class NetworkGeneticSchematic(Container):
    """
    arranges transcription units (TUs) and parts into a grid-based schematic,
    showing interactions between them.
    """

    network: Any
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(background_color="#ffffff", padding=(30, 30, 30, 30))
    )
    layout_orientation: Literal["row", "column"] = "row"
    show_all_tus: bool = False
    grid_gap: Tuple[float, float] = (40.0, 40.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"

    _tu_infos: Dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    _compacted_grid_layout: List[List[Optional[str]]] = PrivateAttr(default_factory=list)
    _interactions: List[Interaction] = PrivateAttr(default_factory=list)
    _tu_components: Dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _part_components: Dict[Tuple[str, str], Component] = PrivateAttr(default_factory=dict)
    _source_map: Dict[Tuple[int, int], str] = PrivateAttr(default_factory=dict)
    _grid_coords: Dict[str, Tuple[int, int]] = PrivateAttr(default_factory=dict)
    _calculated_grid: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def _log_debug(self, message: str, data=None):
        debug_print(self.id or "NetworkGeneticSchematic", message, data)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._process_network_data()

    def _process_network_data(self):
        """extracts info, filters TUs, computes layout grid."""
        if not self.network:
            return
        self._log_debug("processing network data...")
        self._tu_infos = get_tu_informations(self.network)
        raw_grid_layers = get_tu_grid_layout(self.network)
        optimized_layers = optimize_grid_for_source_adjacency(raw_grid_layers, self._tu_infos)
        self._interactions = get_interactions(self.network)

        filtered_layers = []
        if not self.show_all_tus:
            for layer in optimized_layers:
                filtered_layers.append(
                    [
                        tu_id
                        for tu_id in layer
                        if tu_id and tu_id in self._tu_infos and not self._tu_infos[tu_id].is_marker
                    ]
                )
        else:
            filtered_layers = optimized_layers

        transposed_layout = self._transpose_if_needed(filtered_layers)
        self._compact_layout_and_build_maps(transposed_layout)
        self._log_debug(
            f"processed data: {len(self._tu_infos)} TUs, {len(self._grid_coords)} in grid, {len(self._interactions)} interactions."
        )

    def _transpose_if_needed(self, layers: List[List[str]]) -> List[List[Optional[str]]]:
        """transposes layers if orientation is 'column', ensures uniform width."""
        grid = [[cell for cell in row] for row in layers]
        if self.layout_orientation == "column":
            max_inner_len = max((len(sublist) for sublist in grid), default=0)
            padded_layers = [list(layer) + [None] * (max_inner_len - len(layer)) for layer in grid]
            grid = [
                [padded_layers[j][i] for j in range(len(padded_layers))]
                for i in range(max_inner_len)
            ]
        if grid:
            max_len = max((len(row) for row in grid), default=0)
            for row in grid:
                row.extend([None] * (max_len - len(row)))
        return grid

    def _compact_layout_and_build_maps(self, layout: List[List[Optional[str]]]):
        """removes empty rows/columns from layout, builds _grid_coords and _source_map."""
        self._compacted_grid_layout = layout
        if not layout or not layout[0]:
            self._reset_grid_state()
            return

        raw_rows, raw_cols = len(layout), len(layout[0])
        non_empty_rows = {r for r in range(raw_rows) if any(layout[r])}
        non_empty_cols = {c for c in range(raw_cols) if any(layout[r][c] for r in range(raw_rows))}
        if not non_empty_rows or not non_empty_cols:
            self._reset_grid_state()
            return

        row_map = {old_r: new_r for new_r, old_r in enumerate(sorted(list(non_empty_rows)))}
        col_map = {old_c: new_c for new_c, old_c in enumerate(sorted(list(non_empty_cols)))}

        new_num_rows, new_num_cols = len(row_map), len(col_map)
        self._compacted_grid_layout = [[None] * new_num_cols for _ in range(new_num_rows)]
        self._source_map.clear()
        self._grid_coords.clear()

        for r_old, r_new in row_map.items():
            for c_old, c_new in col_map.items():
                tu_id = layout[r_old][c_old]
                if tu_id:
                    self._compacted_grid_layout[r_new][c_new] = tu_id
                    self._grid_coords[tu_id] = (r_new, c_new)
                    s_id = _get_source_id(tu_id, self._tu_infos)
                    self._source_map[(r_new, c_new)] = s_id

        self._calculated_grid.update({"num_rows": new_num_rows, "num_cols": new_num_cols})
        self._log_debug(f"compacted grid: {new_num_rows} rows, {new_num_cols} cols")

    def _reset_grid_state(self):
        """clears internal grid state when layout is empty."""
        self._compacted_grid_layout = []
        self._source_map.clear()
        self._grid_coords.clear()
        self._calculated_grid.update({"num_rows": 0, "num_cols": 0})

    def _create_tu_components(self):
        """creates tu and genetic part components based on processed data."""
        self._tu_components.clear()
        self._part_components.clear()
        for tu_id in self._grid_coords.keys():
            if tu_id not in self._tu_infos:
                logger.warning(f"missing info for tu_id '{tu_id}' in grid, skipping.")
                continue
            tu_info = self._tu_infos[tu_id]
            tu = TranscriptionUnit(id=tu_id, name=tu_info.tu_name)
            tu.parent = self  # set parent link early
            for part in tu_info.parts:
                element = self._create_genetic_element(part, tu_id)
                if element:
                    tu.add_child(element)  # parent link set by add_child
                    self._part_components[(tu_id, part.name)] = element
            self._tu_components[tu_id] = tu

    def _create_genetic_element(self, part_info: PartInfo, tu_id: str) -> Optional[Component]:
        """factory for creating specific geneticpart components."""
        cat, name = part_info.category, part_info.name
        comp_id = f"{tu_id}_{name}"
        el_map = {
            "ERN": (ERN, True),
            "ERN_recog_site_5p": (ERN5pRecog, True),
            "promoter": (Promoter, False),
            "terminator": (Terminator, False),
            "fluo_marker": (FluoMarker, True),
            "uORF_group": (UorfGroup, True),
        }
        if cat == "insulator":
            return None
        el_class, needs_name = el_map.get(cat, (GeneticPart, True))
        kwargs = {"id": comp_id}
        if el_class is GeneticPart:
            kwargs["part_type"] = cat
        if needs_name:
            kwargs["part_name"] = name
        return el_class(**kwargs)

    def _calculate_grid_layout(self, measured_tu_sizes: Dict[str, Size]):
        """calculates grid cell sizes and sets absolute offsets for tus."""
        rows = self._calculated_grid.get("num_rows", 0)
        cols = self._calculated_grid.get("num_cols", 0)
        if rows == 0 or cols == 0:
            return

        col_w = [
            max(
                (
                    measured_tu_sizes[self._compacted_grid_layout[r][c]].width
                    for r in range(rows)
                    if self._compacted_grid_layout[r][c]
                    and self._compacted_grid_layout[r][c] in measured_tu_sizes
                ),
                default=0,
            )
            for c in range(cols)
        ]
        row_h = [
            max(
                (
                    measured_tu_sizes[self._compacted_grid_layout[r][c]].height
                    for c in range(cols)
                    if self._compacted_grid_layout[r][c]
                    and self._compacted_grid_layout[r][c] in measured_tu_sizes
                ),
                default=0,
            )
            for r in range(rows)
        ]
        col_gap, row_gap = self.grid_gap
        col_x = [sum(col_w[:c]) + c * col_gap for c in range(cols)]
        row_y = [sum(row_h[:r]) + r * row_gap for r in range(rows)]

        self._calculated_grid.update(
            {
                "col_widths": col_w,
                "row_heights": row_h,
                "col_x": col_x,
                "row_y": row_y,
                "total_width": sum(col_w) + max(0, cols - 1) * col_gap,
                "total_height": sum(row_h) + max(0, rows - 1) * row_gap,
                "measured_tu_sizes": measured_tu_sizes,
            }
        )

        for tu_id, comp in self._tu_components.items():
            if tu_id in self._grid_coords:
                r, c = self._grid_coords[tu_id]
                # check if tu_id exists in measured sizes (could be missing if it returned None)
                if tu_id not in measured_tu_sizes:
                    logger.warning(f"TU '{tu_id}' missing from measured sizes during grid layout.")
                    continue
                tu_size = measured_tu_sizes[tu_id]
                off_x = (col_w[c] - tu_size.width) / 2 if col_w[c] > tu_size.width else 0
                off_y = (row_h[r] - tu_size.height) / 2 if row_h[r] > tu_size.height else 0
                comp.offset = Offset(absolute=(col_x[c] + off_x, row_y[r] + off_y))
        self._log_debug("calculated grid layout and set tu offsets.")

    def _get_tu_world_bounds(self, r: int, c: int) -> Optional[Tuple[float, float, float, float]]:
        """gets the bounding box of a tu in world coordinates AFTER layout."""
        if r >= len(self._compacted_grid_layout) or c >= len(self._compacted_grid_layout[r]):
            return None
        tu_id = self._compacted_grid_layout[r][c]
        if not tu_id:
            return None
        comp = self._tu_components.get(tu_id)
        if not comp or comp.parent != self:
            return None
        return comp.get_world_bounds()

    def _create_source_annotations(self) -> List[SourceAnnotation]:
        """creates sourceannotation overlays based on calculated tu positions."""
        rows = self._calculated_grid.get("num_rows", 0)
        cols = self._calculated_grid.get("num_cols", 0)
        if rows == 0 or cols == 0 or "col_widths" not in self._calculated_grid:
            return []

        annotations_to_add = []
        covered_coords = set()
        self._log_debug("creating source annotations...")
        for r_start in range(rows):
            for c_start in range(cols):
                start_coord = (r_start, c_start)
                if start_coord in covered_coords or start_coord not in self._source_map:
                    continue
                source_id = self._source_map[start_coord]
                if not source_id:
                    continue  # skip if no source id mapped

                max_width = 0
                for c in range(c_start, cols):
                    if self._source_map.get((r_start, c)) == source_id:
                        max_width += 1
                    else:
                        break
                max_height = 0
                for r_test in range(r_start, rows):
                    if all(
                        self._source_map.get((r_test, c)) == source_id
                        for c in range(c_start, c_start + max_width)
                    ):
                        max_height += 1
                    else:
                        break
                if max_height == 0 or max_width == 0:
                    continue

                r_max, c_max = r_start + max_height - 1, c_start + max_width - 1
                rect_coords = [
                    (r, c) for r in range(r_start, r_max + 1) for c in range(c_start, c_max + 1)
                ]

                # get tu_id for the top-left cell to retrieve source info
                top_left_tu_id = self._compacted_grid_layout[r_start][c_start]
                if not top_left_tu_id or top_left_tu_id not in self._tu_infos:
                    logger.warning(f"Missing TU info for source annotation anchor {top_left_tu_id}")
                    continue
                tu_info = self._tu_infos[top_left_tu_id]

                all_bounds = [
                    self._get_tu_world_bounds(r, c)
                    for r, c in rect_coords
                    if self._compacted_grid_layout[r][c] in self._tu_components
                ]
                valid_bounds = [b for b in all_bounds if b]
                if not valid_bounds:
                    logger.warning(
                        f"No valid TU bounds found for source {source_id} starting at {r_start},{c_start}"
                    )
                    continue

                min_x, min_y = min(b[0] for b in valid_bounds), min(b[1] for b in valid_bounds)
                max_x, max_y = max(b[2] for b in valid_bounds), max(b[3] for b in valid_bounds)

                tag = tu_info.cotx_marker or ""
                if tu_info.aggregation_ratio_label:
                    tag += f"\n{tu_info.aggregation_ratio_label}"

                anno_id = f"anno_{source_id}_{r_start}_{c_start}"
                anno = SourceAnnotation(
                    id=anno_id,
                    parent=self,  # set parent link
                    marker=tu_info.cotx_marker,
                    tag_label=tag.strip(),
                    source_type="plasmid" if tu_info.in_l2 else "cotx",
                )
                jstyle.apply(anno)  # apply styles to get padding etc.
                pad_t, pad_r, pad_b, pad_l = anno.style.padding
                anno_w = (max_x - min_x) + pad_l + pad_r
                anno_h = (max_y - min_y) + pad_t + pad_b
                anno_x, anno_y = min_x - pad_l, min_y - pad_t
                anno.offset = Offset(absolute=(anno_x, anno_y))
                anno._dimensions = Size(width=anno_w, height=anno_h)

                annotations_to_add.append(anno)
                covered_coords.update(rect_coords)
        self._log_debug(f"created {len(annotations_to_add)} source annotations.")
        return annotations_to_add

    def _create_connections(self) -> List[Connection]:
        """creates connection components based on interactions."""
        connections_to_add = []
        curve_map = {
            "orthogonal": OrthogonalCurve,
            "bezier": SimpleBezierCurve,
            "straight": StraightCurve,
        }
        curve_type_cls = curve_map.get(self.connection_style, StraightCurve)

        self._log_debug(
            f"creating {len(self._interactions)} connections (style: {self.connection_style})..."
        )
        for idx, interaction in enumerate(self._interactions):
            src_comp = self._part_components.get((interaction.src_tu_id, interaction.src_part_name))
            tgt_comp = self._part_components.get((interaction.tgt_tu_id, interaction.tgt_part_name))

            if not src_comp or not tgt_comp:
                self._log_debug(
                    f"skipping connection {idx}: missing source ({src_comp is None}) or target ({tgt_comp is None}) components."
                )
                continue

            conn = Connection(
                id=f"connection_{idx}",
                parent=self,  # set parent link
                style_class=[f"connection_{interaction.type}"],
                start_component=src_comp,
                end_component=tgt_comp,
                curve_type=curve_type_cls(),
                end_cap=LineEndFlat(length=7),
                is_overlay=True,
                z_index=-1,
            )
            connections_to_add.append(conn)
        self._log_debug(f"created {len(connections_to_add)} connection components.")
        return connections_to_add

    def measure_and_layout(self, renderer: Optional[BaseRenderer] = None) -> Size:
        """
        overrides container layout to handle grid calculation, overlay creation,
        and final positioning.
        """
        self._log_debug("measure_and_layout start")
        jstyle.apply(self)
        self.children = []

        self._create_tu_components()
        if not self._tu_components:
            self._log_debug("no tu components to layout.")
            return super().measure_and_layout(renderer)  # Call base M&L which returns Size

        measured_tu_sizes = {}
        for tu_id, comp in self._tu_components.items():
            jstyle.apply(comp)  # apply styles before measure
            size_result = comp.measure_and_layout(renderer)
            # check if M&L returned a valid size
            if not isinstance(size_result, Size):
                logger.error(
                    f"measure/layout for tu '{tu_id}' ({type(comp).__name__}) returned {type(size_result)} instead of size. using size(0,0). component state: {comp.model_dump(exclude={'parent','children'}, exclude_none=True, exclude_defaults=True)}"
                )
                size_result = Size()  # use default size if error
            measured_tu_sizes[tu_id] = size_result
        self._log_debug(f"measured {len(measured_tu_sizes)} tu components.")

        self._calculate_grid_layout(measured_tu_sizes)
        # Add TUs to children *after* grid calculation sets their offsets
        self.add_children(self._tu_components.values())
        self._log_debug(f"added {len(self._tu_components)} tu components as children.")

        # --- Measure/Layout base container (including TUs) ---
        # this positions the TUs according to their offsets
        self._log_debug("calling base container measure_and_layout for initial size...")
        base_size = super().measure_and_layout(renderer)
        self._log_debug(f"base measure_and_layout finished, base size: {base_size}")

        # --- Create overlays (Annotations, Connections) ---
        # these need the TUs to be positioned correctly first
        annotations = self._create_source_annotations()
        connections = self._create_connections()
        self.add_children(annotations)
        self.add_children(connections)
        self._log_debug(
            f"added {len(annotations)} annotations and {len(connections)} connections as children."
        )

        # --- Measure/Layout Overlays ---
        # overlays need to be measured after they are added and parent links are set
        self._log_debug("measuring overlays...")
        for overlay_child in annotations + connections:
            overlay_child.measure_and_layout(renderer)

        # --- Final Size Calculation & Content Shifting ---
        all_bounds = [
            c.get_world_bounds()
            for c in self.children
            if c and c.show and c._dimensions.width > 0 and c._dimensions.height > 0
        ]
        valid_bounds = [b for b in all_bounds if b]

        if not valid_bounds:
            min_x, min_y, max_x, max_y = (0, 0, 0, 0)
            self._log_debug("no valid bounds found for final size calculation.")
        else:
            min_x = min(b[0] for b in valid_bounds)
            min_y = min(b[1] for b in valid_bounds)
            max_x = max(b[2] for b in valid_bounds)
            max_y = max(b[3] for b in valid_bounds)

        content_w, content_h = max(0, max_x - min_x), max(0, max_y - min_y)
        pad = self.style.padding
        required_w = content_w + pad[1] + pad[3]
        required_h = content_h + pad[0] + pad[2]

        # use the base size as a minimum, but expand if content requires more
        final_width = max(base_size.width, required_w)
        final_height = max(base_size.height, required_h)
        self._dimensions = Size(width=final_width, height=final_height)
        self._log_debug(f"final schematic dimensions calculated: {self._dimensions}")

        # shift content to align with padding if needed (relative to calculated min_x/min_y)
        off_x, off_y = pad[3] - min_x, pad[0] - min_y
        if abs(off_x) > 1e-6 or abs(off_y) > 1e-6:
            self._log_debug(f"applying content shift: dx={off_x:.2f}, dy={off_y:.2f}")
            # Update only the absolute part of the offset for all children
            for child in self.children:
                child.offset.absolute = (
                    child.offset.absolute[0] + off_x,
                    child.offset.absolute[1] + off_y,
                )
                # Need to recalculate child's matrix? No, render handles it.

        return self._dimensions


def draw_network_schematic(
    network: Any,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
    show_all_tus: bool = False,
    grid_gap: Tuple[float, float] = (40.0, 20.0),
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal",
    layout_orientation: Literal["row", "column"] = "row",
    debug: bool = False,
) -> Tuple[Any, Any, NetworkGeneticSchematic]:
    """
    high-level function to create and render a network genetic schematic.
    """
    schematic = NetworkGeneticSchematic(
        id="network-schematic",
        network=network,
        show_all_tus=show_all_tus,
        grid_gap=grid_gap,
        connection_style=connection_style,
        layout_orientation=layout_orientation,
        debug=debug,
    )
    # delayed imports
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

    renderer = MatplotlibRenderer(debug=debug)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    renderer.render_component(ax, schematic, adjust_lims=True)
    return fig, ax, schematic
