"""
Module for generating network genetic schematics.

Arranges transcription units (TUs) and parts into a grid, showing interactions.
Handles complex TU arrangements by decomposing source annotations into minimal
rectangles to avoid overlaps.
"""

from typing import Any, Literal
from pydantic import Field, PrivateAttr, BaseModel
import numpy as np
from collections import defaultdict

from jeanplot.core.component import Component
from jeanplot.core.container import Container
from jeanplot.core.models import Size, BoxStyle, LayoutConstraints, Offset
from jeanplot.gene.elements import (
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
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.core.svg import LineEndFlat
from jeanplot._deprecated.network_utils import (
    get_tu_informations,
    get_tu_grid_layout,
    get_interactions,
    optimize_grid_for_source_adjacency,
    _get_source_id,
    TUInfo,
    Interaction,
    PartInfo,
)
from jeanplot.core.style import jstyle
from jeanplot.core.debug import get_logger
from jeanplot.core.renderer import BaseRenderer


logger = get_logger(__name__)


class RectangleInfo(BaseModel):
    """Stores the grid boundaries of a calculated annotation rectangle."""

    r_min: int
    c_min: int
    r_max: int
    c_max: int


class SourceAnnotation(Container):
    """visual bounding box and tag for a source group (or part of one) in the grid."""

    is_overlay: bool = True
    source_id: str  # track which source this annotation belongs to
    has_tag: bool = False  # only one annotation per source gets the tag
    source_type: Literal["plasmid", "cotx"] | None = "cotx"
    marker: str | None = None
    tag_label: str | None = None
    style_class: list[str] = ["source_annotation"]
    _source_proxy: Source | None = PrivateAttr(default=None)
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )
    style: BoxStyle = Field(default_factory=BoxStyle)

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        # only create the tag container if this specific annotation instance should display it
        if self.has_tag:
            self._source_proxy = Source(
                id=f"proxy_{self.id}" if self.id else None,
                source_type=self.source_type,
                marker=self.marker,
                tag_label=self.tag_label,
                children=[],
                parent=self,
            )
            # ensure proxy gets styled early
            jstyle.apply(self._source_proxy)
            tag_cont = getattr(self._source_proxy, "_tag_container", None)
            if tag_cont:
                # detach tag container from proxy and attach to annotation itself
                if tag_cont in self._source_proxy.children:
                    self._source_proxy.children.remove(tag_cont)
                tag_cont.parent = self
                self.add_child(tag_cont)

    def render(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        """render the border and the tag (if applicable)."""
        if not self.show:
            return

        # render border if styled
        if (
            self.style.border_width > 0
            and self.style.border_color
            and self.style.border_color.lower() != "none"
        ):
            border_render_style = self.style.model_copy(
                update={"background_color": None, "shadow": None}
            )
            if self._dimensions.width > 0 and self._dimensions.height > 0:
                renderer.render_rectangle(
                    context, self._dimensions, border_render_style, matrix, component=self
                )

        if self.debug:
            renderer.render_debug(context, self, matrix)

        # render children (i.e., the tag container if has_tag is true)
        Container.render(self, renderer, context, matrix)


class SchematicComponentFactory:
    """Creates components for the NetworkGeneticSchematic."""

    def __init__(
        self, tu_infos: dict[str, TUInfo], interactions: list[Interaction], connection_style: str
    ):
        self._tu_infos = tu_infos
        self._interactions = interactions
        self._connection_style = connection_style
        self._part_components_cache: dict[tuple[str, str], Component] = {}

    def _create_genetic_element(self, part_info: PartInfo, tu_id: str) -> Component | None:
        """factory for creating specific geneticpart components."""
        cat, name = part_info.category, part_info.name
        comp_id = f"{tu_id}_{name}"

        # map category to class and whether name is needed for constructor/styling
        el_map: dict[str, tuple[type[GeneticPart], bool]] = {
            "ERN": (ERN, True),
            "ERN_recog_site_5p": (ERN5pRecog, True),
            "promoter": (Promoter, False),
            "terminator": (Terminator, False),
            "fluo_marker": (FluoMarker, True),
            "uORF_group": (UorfGroup, True),
        }

        if cat == "insulator":
            return None  # skip insulators

        el_class, needs_name = el_map.get(cat, (GeneticPart, True))

        kwargs = {"id": comp_id}
        if el_class is GeneticPart:  # handle generic parts if needed
            kwargs["part_type"] = cat
        if needs_name:
            kwargs["part_name"] = name

        element = el_class(**kwargs)
        self._part_components_cache[(tu_id, name)] = element  # cache for connection lookup
        return element

    def create_tu(self, tu_id: str) -> TranscriptionUnit | None:
        """creates a transcription unit component with its child parts."""
        if tu_id not in self._tu_infos:
            logger.warning(f"tu_id '{tu_id}' not found in tu_infos during TU creation.")
            return None
        tu_info = self._tu_infos[tu_id]

        tu = TranscriptionUnit(id=tu_id, name=tu_info.tu_name)
        for part in tu_info.parts:
            element = self._create_genetic_element(part, tu_id)
            if element:
                tu.add_child(element)
        return tu

    def create_connection(
        self, interaction: Interaction, index: int, parent: Component
    ) -> Connection | None:
        """creates a connection component representing an interaction."""
        curve_map = {
            "orthogonal": OrthogonalCurve,
            "bezier": SimpleBezierCurve,
            "straight": StraightCurve,
        }
        curve_type_cls = curve_map.get(self._connection_style, StraightCurve)

        # lookup cached part components using (tu_id, part_name) tuples
        src_key = (interaction.src_tu_id, interaction.src_part_name)
        tgt_key = (interaction.tgt_tu_id, interaction.tgt_part_name)
        src_comp = self._part_components_cache.get(src_key)
        tgt_comp = self._part_components_cache.get(tgt_key)

        if not src_comp:
            logger.warning(f"source component not found for connection: {src_key}")
            return None
        if not tgt_comp:
            logger.warning(f"target component not found for connection: {tgt_key}")
            return None

        return Connection(
            id=f"connection_{index}",
            parent=parent,
            style_class=[
                f"connection_{interaction.type}",
                f"connection_{interaction.type}_{interaction.src_part_name}",
            ],
            start_component=src_comp,
            end_component=tgt_comp,
            curve_type=curve_type_cls(),
            end_cap=LineEndFlat(),  # perhaps make configurable?
            is_overlay=True,  # connections don't affect layout
            z_index=10,  # render above TUs/annotations
            auto_route=True,  # use anchors if available
        )


class NetworkGeneticSchematic(Container):
    """
    arranges transcription units (TUs) and parts into a grid-based schematic,
    showing interactions between them. uses a factory for component creation
    and positions components absolutely based on grid calculations. Handles
    non-rectangular source groupings by creating multiple annotation boxes.
    """

    network: Any
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(30, 30, 30, 30)))
    layout_orientation: Literal["row", "column"] = "column"
    show_all_tus: bool = False
    grid_gap: tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    node_type: Literal["translation", "transcription"] = "translation"
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )

    # --- Internal State ---
    _tu_infos: dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    _compacted_grid_layout: list[list[str | None]] = PrivateAttr(default_factory=list)
    _interactions: list[Interaction] = PrivateAttr(default_factory=list)
    _grid_coords: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)
    _source_rectangles: dict[str, list[RectangleInfo]] = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )
    _top_left_tu_per_source: dict[str, str] = PrivateAttr(default_factory=dict)

    _component_factory: SchematicComponentFactory | None = PrivateAttr(default=None)
    _tu_components: dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _annotation_components: list[SourceAnnotation] = PrivateAttr(default_factory=list)
    _connection_components: list[Connection] = PrivateAttr(default_factory=list)

    _grid_dimensions: dict[str, Any] = PrivateAttr(default_factory=dict)
    _content_offset: tuple[float, float] = PrivateAttr(default=(0.0, 0.0))

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        self._process_network_data()
        self._build_components()

    def _process_network_data(self):
        """extracts info, filters TUs, computes layout grid, decomposes sources."""
        if not self.network:
            self._log_debug("network object is missing, cannot process data.")
            return
        self._log_debug("processing network data...")
        self._tu_infos = get_tu_informations(self.network)
        raw_grid_layers = get_tu_grid_layout(self.network, node_type=self.node_type)

        # Find TUs that are missing from the grid (those without translation nodes)
        all_tus = set(self._tu_infos.keys())
        tus_in_grid = set()
        for layer in raw_grid_layers:
            tus_in_grid.update(layer)
        missing_tus = all_tus - tus_in_grid

        # Add missing TUs to appropriate layers based on their connections
        if missing_tus and raw_grid_layers:
            # For now, add missing TUs to the first layer where they have connections
            # In this case, x2_a+ and b_a- should be in layer 0 with other source TUs
            self._log_debug(f"Adding missing TUs to grid: {missing_tus}")
            # Add to first layer (they're all source-level TUs)
            raw_grid_layers[0].extend(sorted(missing_tus))

        optimized_layers = optimize_grid_for_source_adjacency(raw_grid_layers, self._tu_infos)
        self._interactions = get_interactions(self.network)

        # filter TUs based on show_all_tus flag
        filtered_layers = [
            [
                tu_id
                for tu_id in layer
                if tu_id
                and tu_id in self._tu_infos
                and (self.show_all_tus or not self._tu_infos[tu_id].is_marker)
            ]
            for layer in optimized_layers
        ]

        transposed_layout = self._transpose_if_needed(filtered_layers)
        self._compact_layout_and_build_maps(transposed_layout)
        self._decompose_source_groups_into_rectangles()
        self._log_debug(
            f"processed data: {len(self._grid_coords)} TUs, {len(self._source_rectangles)} sources ({sum(len(r) for r in self._source_rectangles.values())} rects), {len(self._interactions)} interactions."
        )

    def _transpose_if_needed(self, layers: list[list[str]]) -> list[list[str | None]]:
        """transposes layers if orientation is 'column', ensures uniform width."""
        if not layers:
            return []
        grid = [[cell for cell in row] for row in layers]
        if self.layout_orientation == "column":
            max_inner_len = max((len(sublist) for sublist in grid), default=0)
            if max_inner_len == 0:
                return []
            # pad rows to max length before transposing
            padded_layers = [list(layer) + [None] * (max_inner_len - len(layer)) for layer in grid]
            # transpose
            grid = [
                [padded_layers[j][i] for j in range(len(padded_layers))]
                for i in range(max_inner_len)
            ]
        # ensure final grid is rectangular
        if grid:
            max_len = max((len(row) for row in grid), default=0)
            for row in grid:
                row.extend([None] * (max_len - len(row)))
        return grid

    def _compact_layout_and_build_maps(self, layout: list[list[str | None]]):
        """removes empty rows/columns, builds _grid_coords and _source_groups."""
        self._grid_coords.clear()
        source_coords: dict[str, list[tuple[int, int]]] = defaultdict(list)

        if not layout or not layout[0]:
            self._compacted_grid_layout = []
            self._grid_dimensions = {"num_rows": 0, "num_cols": 0}
            return

        raw_rows, raw_cols = len(layout), len(layout[0])
        non_empty_rows = {r for r in range(raw_rows) if any(layout[r])}
        non_empty_cols = {c for c in range(raw_cols) if any(layout[r][c] for r in range(raw_rows))}

        if not non_empty_rows or not non_empty_cols:
            self._compacted_grid_layout = []
            self._grid_dimensions = {"num_rows": 0, "num_cols": 0}
            return

        row_map = {old_r: new_r for new_r, old_r in enumerate(sorted(list(non_empty_rows)))}
        col_map = {old_c: new_c for new_c, old_c in enumerate(sorted(list(non_empty_cols)))}

        new_num_rows, new_num_cols = len(row_map), len(col_map)
        self._compacted_grid_layout = [[None] * new_num_cols for _ in range(new_num_rows)]
        self._grid_dimensions.update({"num_rows": new_num_rows, "num_cols": new_num_cols})

        for r_old, r_new in row_map.items():
            for c_old, c_new in col_map.items():
                tu_id = layout[r_old][c_old]
                if tu_id and tu_id in self._tu_infos:
                    self._compacted_grid_layout[r_new][c_new] = tu_id
                    grid_coord = (r_new, c_new)
                    self._grid_coords[tu_id] = grid_coord
                    s_id = _get_source_id(tu_id, self._tu_infos)
                    source_coords[s_id].append(grid_coord)

        # identify the top-leftmost tu for each source (used for placing the tag)
        self._top_left_tu_per_source.clear()
        for s_id, coords in source_coords.items():
            if coords:
                min_r, min_c = min(coords, key=lambda x: (x[0], x[1]))
                top_left_tu_id = self._compacted_grid_layout[min_r][min_c]
                if top_left_tu_id:
                    self._top_left_tu_per_source[s_id] = top_left_tu_id

        self._log_debug(
            f"compacted grid: {new_num_rows} rows, {new_num_cols} cols. Found {len(source_coords)} source groups."
        )
        # store source coords temporarily for decomposition
        self._source_groups_coords = source_coords  # temporary storage

    def _decompose_source_groups_into_rectangles(self):
        """
        uses the iterative maximal rectangle finding algorithm to cover all TUs
        of a source with the minimum number of non-overlapping rectangles.
        """
        self._source_rectangles.clear()
        source_coords = getattr(self, "_source_groups_coords", {})
        if not source_coords:
            return

        for source_id, coords_list in source_coords.items():
            source_coords_set = set(coords_list)
            covered_coords = set()
            rectangles: list[RectangleInfo] = []

            while covered_coords != source_coords_set:
                # find top-leftmost uncovered coordinate
                uncovered = sorted(
                    list(source_coords_set - covered_coords), key=lambda x: (x[0], x[1])
                )
                if not uncovered:
                    break  # Should not happen if loop condition is correct
                r_start, c_start = uncovered[0]

                # find maximal rectangle starting from (r_start, c_start)
                r_end = r_start
                c_end = c_start

                # expand right
                while True:
                    next_c = c_end + 1
                    can_expand_col = True
                    for r_check in range(r_start, r_end + 1):
                        if (r_check, next_c) not in source_coords_set:
                            can_expand_col = False
                            break
                    if can_expand_col:
                        c_end = next_c
                    else:
                        break

                # expand down
                while True:
                    next_r = r_end + 1
                    can_expand_row = True
                    for c_check in range(c_start, c_end + 1):
                        if (next_r, c_check) not in source_coords_set:
                            can_expand_row = False
                            break
                    if can_expand_row:
                        r_end = next_r
                    else:
                        break

                # store rectangle and mark covered coords
                rect = RectangleInfo(r_min=r_start, c_min=c_start, r_max=r_end, c_max=c_end)
                rectangles.append(rect)
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        covered_coords.add((r, c))

            self._source_rectangles[source_id] = rectangles
            self._log_debug(f"decomposed source '{source_id}' into {len(rectangles)} rectangles.")
        del self._source_groups_coords  # remove temporary storage

    def _build_components(self):
        """creates TU, Annotation, and Connection instances using the factory."""
        if not self._tu_infos:
            self._log_debug("no tu_infos available, cannot build components.")
            return

        self._component_factory = SchematicComponentFactory(
            self._tu_infos, self._interactions, self.connection_style
        )
        self._tu_components.clear()
        self._annotation_components.clear()
        self._connection_components.clear()

        for tu_id in self._grid_coords.keys():
            tu_comp = self._component_factory.create_tu(tu_id)
            if tu_comp:
                self._tu_components[tu_id] = tu_comp

        for i, interaction in enumerate(self._interactions):
            conn_comp = self._component_factory.create_connection(interaction, i, self)
            if conn_comp:
                self._connection_components.append(conn_comp)

        source_groups_map: dict[str, list[str]] = defaultdict(list)
        for tu_id in self._tu_infos:
            s_id = _get_source_id(tu_id, self._tu_infos)
            source_groups_map[s_id].append(tu_id)

        self._log_debug("Rebuilt Source Groups Map:", source_groups_map)
        self._log_debug("Top-Left TU per Source Map:", self._top_left_tu_per_source)

        for source_id, rectangles in self._source_rectangles.items():
            marker_tu_info: TUInfo | None = None
            source_tu_ids = source_groups_map.get(source_id, [])
            if not source_tu_ids:
                self._log_debug(
                    f"Warning: No TUs found for source_id '{source_id}'. Skipping annotation."
                )
                continue

            for tid in source_tu_ids:
                if tid in self._tu_infos and self._tu_infos[tid].is_marker:
                    marker_tu_info = self._tu_infos[tid]
                    self._log_debug(
                        f"Found marker TU '{tid}' for source '{source_id}'. Marker: {marker_tu_info.cotx_marker}"
                    )
                    break

            top_left_tu_id = self._top_left_tu_per_source.get(source_id)
            tag_label_tu_info: TUInfo | None = None
            if top_left_tu_id and top_left_tu_id in self._tu_infos:
                tag_label_tu_info = self._tu_infos[top_left_tu_id]
            else:
                first_tu_id = source_tu_ids[0] if source_tu_ids else None
                if first_tu_id and first_tu_id in self._tu_infos:
                    tag_label_tu_info = self._tu_infos[first_tu_id]
                    self._log_debug(
                        f"Warning: top_left_tu_id '{top_left_tu_id}' not found/valid for source '{source_id}'. Using first TU '{first_tu_id}' for tag label info."
                    )
                else:
                    self._log_debug(
                        f"Error: Cannot determine any TUInfo for tag label for source '{source_id}'. Skipping annotation."
                    )
                    continue

            # determine source type from the tag_label_tu_info
            annotation_source_type = "plasmid" if tag_label_tu_info.in_l2 else "cotx"
            # determine marker for styling from the marker_tu_info
            annotation_marker = marker_tu_info.cotx_marker if marker_tu_info else None

            # Use cotx_name if available, otherwise fall back to marker name
            tag_str_parts = []
            if tag_label_tu_info.cotx_name:
                # If we have a cotx name, use it as the primary label
                tag_str_parts.append(tag_label_tu_info.cotx_name)
            elif annotation_marker:
                # Otherwise use the marker name
                tag_str_parts.append(annotation_marker)
            if tag_label_tu_info.aggregation_ratio_label:
                tag_str_parts.append(tag_label_tu_info.aggregation_ratio_label)
            tag_str = "\n".join(tag_str_parts)

            self._log_debug(f"--- Creating Annotation for Source: {source_id} ---")
            self._log_debug(f"  Source TU IDs: {source_tu_ids}")
            self._log_debug(f"  Top-Left TU ID for Tag Logic: {top_left_tu_id}")
            self._log_debug(
                f"  Tag Label TU Info: {tag_label_tu_info.tu_id if tag_label_tu_info else 'None'}"
            )
            self._log_debug(
                f"  Marker TU Info: {marker_tu_info.tu_id if marker_tu_info else 'None'}"
            )
            self._log_debug(f"  Annotation Source Type: {annotation_source_type}")
            self._log_debug(f"  Annotation Marker (for styling): {annotation_marker}")
            self._log_debug(f"  Annotation Tag Label (for display): '{tag_str}'")
            self._log_debug(f"  Number of Rectangles: {len(rectangles)}")

            for i, rect in enumerate(rectangles):
                has_tag = False
                if top_left_tu_id:
                    top_left_r, top_left_c = self._grid_coords.get(top_left_tu_id, (-1, -1))
                    if top_left_r != -1 and (
                        rect.r_min <= top_left_r <= rect.r_max
                        and rect.c_min <= top_left_c <= rect.c_max
                    ):
                        has_tag = True
                elif i == 0:
                    has_tag = True
                    self._log_debug(
                        f"Assigning tag to first rectangle (index 0) for source {source_id} due to missing top-left TU."
                    )

                anno_id = f"anno_{source_id}_{rect.r_min}_{rect.c_min}"
                self._log_debug(
                    f"  Creating Rect {i + 1}/{len(rectangles)}: ID={anno_id}, HasTag={has_tag}, TagLabel='{tag_str if has_tag else ''}'"
                )

                anno = SourceAnnotation(
                    id=anno_id,
                    parent=self,
                    source_id=source_id,
                    has_tag=has_tag,
                    marker=annotation_marker,
                    tag_label=tag_str,
                    source_type=annotation_source_type,
                    z_index=-5,
                )
                self._annotation_components.append(anno)

        self._log_debug(
            f"built components: {len(self._tu_components)} TUs, {len(self._annotation_components)} Annotations, {len(self._connection_components)} Connections"
        )

    def _calculate_grid_cell_dims(self, measured_tu_sizes: dict[str, Size]):
        """calculates the required width/height for each grid cell."""
        rows = self._grid_dimensions.get("num_rows", 0)
        cols = self._grid_dimensions.get("num_cols", 0)
        if rows == 0 or cols == 0:
            return [], []

        col_widths = [0.0] * cols
        row_heights = [0.0] * rows
        for tu_id, (r, c) in self._grid_coords.items():
            size = measured_tu_sizes.get(tu_id)
            if size:
                col_widths[c] = max(col_widths[c], size.width)
                row_heights[r] = max(row_heights[r], size.height)
        return col_widths, row_heights

    def _position_components(self, measured_tu_sizes: dict[str, Size]):
        """calculates grid dimensions and sets absolute offsets for TUs and Annotations."""
        col_widths, row_heights = self._calculate_grid_cell_dims(measured_tu_sizes)
        rows = self._grid_dimensions.get("num_rows", 0)
        cols = self._grid_dimensions.get("num_cols", 0)
        if not col_widths or not row_heights:
            self._log_debug("grid dimensions are empty, cannot position components.")
            self._grid_dimensions = {}
            return

        col_gap, row_gap = self.grid_gap
        col_x = [sum(col_widths[:c]) + c * col_gap for c in range(cols)]
        row_y = [sum(row_heights[:r]) + r * row_gap for r in range(rows)]
        total_width = sum(col_widths) + max(0, cols - 1) * col_gap
        total_height = sum(row_heights) + max(0, rows - 1) * row_gap

        self._grid_dimensions.update(
            {
                "col_widths": col_widths,
                "row_heights": row_heights,
                "col_x": col_x,
                "row_y": row_y,
                "total_width": total_width,
                "total_height": total_height,
            }
        )

        # position TUs centered within their cells
        for tu_id, comp in self._tu_components.items():
            if tu_id in self._grid_coords:
                r, c = self._grid_coords[tu_id]
                tu_size = measured_tu_sizes.get(tu_id)
                if not tu_size:
                    continue
                cell_w, cell_h = col_widths[c], row_heights[r]
                off_x = (cell_w - tu_size.width) / 2.0
                off_y = (cell_h - tu_size.height) / 2.0
                comp.offset = Offset(absolute=(col_x[c] + off_x, row_y[r] + off_y))
                comp.parent = self

        # position Annotations based on their rectangle bounds
        for anno in self._annotation_components:
            rectangles = self._source_rectangles.get(anno.source_id, [])
            # find the rectangle this annotation corresponds to (based on id suffix)
            rect_r_min, rect_c_min = -1, -1
            try:
                parts = anno.id.split("_")
                rect_r_min, rect_c_min = int(parts[-2]), int(parts[-1])
            except (IndexError, ValueError):
                logger.warning(f"could not parse rectangle coords from annotation id: {anno.id}")
                continue

            target_rect = next(
                (r for r in rectangles if r.r_min == rect_r_min and r.c_min == rect_c_min),
                None,
            )
            if not target_rect:
                logger.warning(f"rectangle not found for annotation {anno.id}")
                continue

            # calculate bounding box in local coords based on grid cells
            local_min_x = col_x[target_rect.c_min]
            local_min_y = row_y[target_rect.r_min]
            local_max_x = col_x[target_rect.c_max] + col_widths[target_rect.c_max]
            local_max_y = row_y[target_rect.r_max] + row_heights[target_rect.r_max]

            jstyle.apply(anno)  # ensure padding is applied from styles
            pad_t, pad_r, pad_b, pad_l = anno.style.padding

            anno_w = (local_max_x - local_min_x) + pad_l + pad_r
            anno_h = (local_max_y - local_min_y) + pad_t + pad_b
            anno_x_local = local_min_x - pad_l
            anno_y_local = local_min_y - pad_t

            # set size and position
            anno.min_dimensions = Size(width=anno_w, height=anno_h)
            anno.max_dimensions = Size(width=anno_w, height=anno_h)
            anno.offset = Offset(absolute=(anno_x_local, anno_y_local))
            anno.parent = self
            # self._log_debug(f"positioned annotation {anno.id}: offset={anno.offset}, size={anno.min_dimensions}")

        self._log_debug("calculated grid dimensions and set component offsets.")

    def _calculate_content_bounds(self) -> tuple[float, float, float, float]:
        """calculates the min/max local coordinates occupied by all children."""
        if not self.children:
            return (0, 0, 0, 0)
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        found = False
        for c in self.children:
            if (
                c
                and c.show
                and hasattr(c, "_dimensions")
                and hasattr(c, "offset")
                and isinstance(c.offset, Offset)
                and c._dimensions.width >= 0
                and c._dimensions.height >= 0
            ):
                ox, oy = c.offset.absolute  # use absolute offset as component origin
                w, h = c._dimensions.width, c._dimensions.height
                min_x = min(min_x, ox)
                min_y = min(min_y, oy)
                max_x = max(max_x, ox + w)
                max_y = max(max_y, oy + h)
                found = True
        return (min_x, min_y, max_x, max_y) if found else (0, 0, 0, 0)

    def measure_and_layout(self, renderer: BaseRenderer | None = None) -> Size:
        """measures and lays out the schematic components absolutely."""
        self._log_debug("measure_and_layout start")
        jstyle.apply(self)  # apply style to self first
        self.children = []  # clear children from previous runs
        self._content_offset = (0.0, 0.0)

        # check if components were built
        if not self._component_factory or not self._tu_components:
            self._log_debug("components not built, cannot layout.")
            self._dimensions = self._apply_constraints(Size())
            return self._dimensions

        # --- Measurement Phase ---
        measured_tu_sizes: dict[str, Size] = {}
        for tu_id, comp in self._tu_components.items():
            jstyle.apply(comp)
            # TUs measure naturally based on their parts
            size = comp.measure_and_layout(renderer)
            measured_tu_sizes[tu_id] = size if isinstance(size, Size) else Size()

        # --- Positioning Phase ---
        self._position_components(measured_tu_sizes)
        if not self._grid_dimensions:  # check if positioning failed
            self._log_debug("grid dimensions not calculated, aborting layout.")
            self._dimensions = self._apply_constraints(Size())
            return self._dimensions

        # annotations size/pos are now set, measure them (mostly for their tag)
        for anno in self._annotation_components:
            jstyle.apply(anno)
            anno.measure_and_layout(renderer)

        # measure connections (important for their bounding box if debug enabled)
        for conn in self._connection_components:
            jstyle.apply(conn)
            conn.measure_and_layout(renderer)

        # --- Final Sizing & Shift ---
        # Add all components to children list (defines render order & content bounds)
        self.add_children(self._annotation_components)  # annotations behind tus
        self.add_children(self._tu_components.values())
        self.add_children(self._connection_components)  # connections above all

        # calculate content bounds based on final absolute positions/sizes
        min_x, min_y, max_x, max_y = self._calculate_content_bounds()

        # determine final schematic size including padding
        pad_t, pad_r, pad_b, pad_l = self.style.padding
        content_w = max(0, max_x - min_x)
        content_h = max(0, max_y - min_y)
        req_w = content_w + pad_l + pad_r
        req_h = content_h + pad_t + pad_b

        # apply constraints to get final size
        self._natural_dimensions = Size(width=req_w, height=req_h)
        self._dimensions = self._apply_constraints(self._natural_dimensions)

        # calculate shift needed to center content within padding
        shift_x = pad_l - min_x
        shift_y = pad_t - min_y
        self._content_offset = (shift_x, shift_y)

        # apply shift to all children's absolute offsets
        if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
            self._log_debug(f"applying content shift: dx={shift_x:.2f}, dy={shift_y:.2f}")
            for child in self.children:
                if hasattr(child, "offset") and isinstance(child.offset, Offset):
                    orig_abs = child.offset.absolute
                    child.offset.absolute = (orig_abs[0] + shift_x, orig_abs[1] + shift_y)

        self._log_debug(f"schematic layout complete, final dimensions: {self._dimensions}")
        return self._dimensions

    # override _layout_children to do nothing, as positioning is absolute
    def _layout_children(self, renderer: BaseRenderer | None):
        # we need to ensure child containers (like TU lines) get their layout pass called
        for child in self.children:
            if isinstance(child, Container):
                # trigger their internal layout mechanism if needed
                child._layout_children(renderer)
        pass  # prevent default container layout logic
