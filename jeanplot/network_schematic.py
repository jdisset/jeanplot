"""
Schematic drawing logic.
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from pydantic import Field, PrivateAttr, model_validator
import logging
import numpy as np
from collections import defaultdict

# use absolute imports
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
from jeanplot.debug import debug_print, get_logger
from jeanplot.renderer import BaseRenderer

logger = get_logger(__name__)


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
    style: BoxStyle = Field(default_factory=BoxStyle)

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        self._source_proxy = Source(
            id=f"proxy_{self.id}" if self.id else None,
            source_type=self.source_type,
            marker=self.marker,
            tag_label=self.tag_label,
            children=[],
            parent=self,
        )
        jstyle.apply(self._source_proxy)
        tag_cont = getattr(self._source_proxy, "_tag_container", None)
        if tag_cont:
            if tag_cont in self._source_proxy.children:
                self._source_proxy.children.remove(tag_cont)
            tag_cont.parent = self
            self.add_child(tag_cont)

    def render(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        if not self.show:
            return
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
        Container.render(self, renderer, context, matrix)  # render children (tag)

    def _log_debug(self, message: str, data=None):
        debug_print(self.id or "SourceAnnotation", message, data)


class SchematicComponentFactory:
    """Creates components for the NetworkGeneticSchematic."""

    def __init__(
        self, tu_infos: Dict[str, TUInfo], interactions: List[Interaction], connection_style: str
    ):
        self._tu_infos = tu_infos
        self._interactions = interactions
        self._connection_style = connection_style
        self._part_components_cache: Dict[Tuple[str, str], Component] = {}

    def _create_genetic_element(self, part_info: PartInfo, tu_id: str) -> Optional[Component]:
        """factory for creating specific geneticpart components."""
        # (Same logic as the previous _create_genetic_element method)
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
        element = el_class(**kwargs)
        self._part_components_cache[(tu_id, name)] = element  # cache part component
        return element

    def create_tu(self, tu_id: str) -> Optional[TranscriptionUnit]:
        """Creates a TU component with its parts."""
        if tu_id not in self._tu_infos:
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
    ) -> Optional[Connection]:
        """Creates a Connection component for an interaction."""
        curve_map = {
            "orthogonal": OrthogonalCurve,
            "bezier": SimpleBezierCurve,
            "straight": StraightCurve,
        }
        curve_type_cls = curve_map.get(self._connection_style, StraightCurve)

        src_key = (interaction.src_tu_id, interaction.src_part_name)
        tgt_key = (interaction.tgt_tu_id, interaction.tgt_part_name)
        src_comp = self._part_components_cache.get(src_key)
        tgt_comp = self._part_components_cache.get(tgt_key)

        if not src_comp or not tgt_comp:
            return None

        return Connection(
            id=f"connection_{index}",
            parent=parent,
            style_class=[f"connection_{interaction.type}"],
            start_component=src_comp,
            end_component=tgt_comp,
            curve_type=curve_type_cls(),
            end_cap=LineEndFlat(),
            is_overlay=True,
            z_index=10,
            auto_route=True,
        )

    def get_part_component(self, tu_id: str, part_name: str) -> Optional[Component]:
        return self._part_components_cache.get((tu_id, part_name))

    def get_all_part_components(self) -> Dict[Tuple[str, str], Component]:
        return self._part_components_cache


class NetworkGeneticSchematic(Container):
    """
    arranges transcription units (TUs) and parts into a grid-based schematic,
    showing interactions between them. Uses a factory for component creation
    and positions components absolutely based on grid calculations.
    """

    network: Any
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(background_color="#ffffff", padding=(30, 30, 30, 30))
    )
    layout_orientation: Literal["row", "column"] = "row"
    show_all_tus: bool = False
    grid_gap: Tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    # Layout constraints of the container itself are mostly ignored, as children are placed absolutely
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )

    # --- Processed Data (set in model_post_init) ---
    _tu_infos: Dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    _compacted_grid_layout: List[List[Optional[str]]] = PrivateAttr(default_factory=list)
    _interactions: List[Interaction] = PrivateAttr(default_factory=list)
    _grid_coords: Dict[str, Tuple[int, int]] = PrivateAttr(default_factory=dict)
    _source_groups: Dict[str, List[Tuple[int, int]]] = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )  # source_id -> list of (r, c)

    # --- Component Storage (built early, positioned later) ---
    _component_factory: Optional[SchematicComponentFactory] = PrivateAttr(default=None)
    _tu_components: Dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _annotation_components: Dict[str, SourceAnnotation] = PrivateAttr(default_factory=dict)
    _connection_components: List[Connection] = PrivateAttr(default_factory=list)

    # --- Layout State ---
    _grid_dimensions: Dict[str, Any] = PrivateAttr(default_factory=dict)  # Stores col_w, row_h etc.
    _content_offset: Tuple[float, float] = PrivateAttr(
        default=(0.0, 0.0)
    )  # Shift needed for padding

    def _log_debug(self, message: str, data=None):
        debug_print(self.id or "NetworkGeneticSchematic", message, data)

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        self._process_network_data()  # extract info and determine grid structure
        self._build_components()  # create component instances via factory

    def _process_network_data(self):
        """extracts info, filters tus, computes layout grid structure and source groups."""
        if not self.network:
            return
        self._log_debug("processing network data...")
        self._tu_infos = get_tu_informations(self.network)
        raw_grid_layers = get_tu_grid_layout(self.network)
        optimized_layers = optimize_grid_for_source_adjacency(raw_grid_layers, self._tu_infos)
        self._interactions = get_interactions(self.network)

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
        self._compact_layout_and_build_maps(
            transposed_layout
        )  # populates grid, coords, source_groups
        self._log_debug(
            f"processed data: {len(self._grid_coords)} TUs in grid, {len(self._source_groups)} source groups, {len(self._interactions)} interactions."
        )

    def _transpose_if_needed(self, layers: List[List[str]]) -> List[List[Optional[str]]]:
        """transposes layers if orientation is 'column', ensures uniform width."""
        # (Same logic as before)
        grid = [[cell for cell in row] for row in layers]
        if self.layout_orientation == "column":
            max_inner_len = max((len(sublist) for sublist in grid), default=0)
            if max_inner_len == 0:
                return []
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
        """removes empty rows/columns, builds _grid_coords and _source_groups."""
        # (Modified to build _source_groups)
        self._compacted_grid_layout = layout
        self._grid_dimensions = {"num_rows": 0, "num_cols": 0}
        self._grid_coords.clear()
        self._source_groups.clear()

        if not layout or not layout[0]:
            return

        raw_rows, raw_cols = len(layout), len(layout[0])
        non_empty_rows = {r for r in range(raw_rows) if any(layout[r])}
        non_empty_cols = {c for c in range(raw_cols) if any(layout[r][c] for r in range(raw_rows))}
        if not non_empty_rows or not non_empty_cols:
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
                    self._grid_coords[tu_id] = (r_new, c_new)
                    s_id = _get_source_id(tu_id, self._tu_infos)
                    self._source_groups[s_id].append((r_new, c_new))  # Store coords per source_id

        self._log_debug(
            f"compacted grid: {new_num_rows} rows, {new_num_cols} cols. Found {len(self._source_groups)} source groups."
        )

    def _build_components(self):
        """Creates TU, Annotation, and Connection instances using the factory."""
        if not self._tu_infos:
            return  # nothing to build if no data

        self._component_factory = SchematicComponentFactory(
            self._tu_infos, self._interactions, self.connection_style
        )
        self._tu_components.clear()
        self._annotation_components.clear()
        self._connection_components.clear()

        # Build TUs
        for tu_id in self._grid_coords.keys():
            tu_comp = self._component_factory.create_tu(tu_id)
            if tu_comp:
                self._tu_components[tu_id] = tu_comp

        # Build Connections
        for i, interaction in enumerate(self._interactions):
            conn_comp = self._component_factory.create_connection(interaction, i, self)
            if conn_comp:
                self._connection_components.append(conn_comp)

        # Build Annotation Placeholders (positioning/sizing happens later)
        # We create them now so styling can be applied early if needed
        for source_id, group_coords in self._source_groups.items():
            if not group_coords:
                continue
            # Get info from top-left TU
            r_start, c_start = min(group_coords, key=lambda x: (x[0], x[1]))
            top_left_tu_id = self._compacted_grid_layout[r_start][c_start]
            if not top_left_tu_id or top_left_tu_id not in self._tu_infos:
                continue
            tu_info = self._tu_infos[top_left_tu_id]

            tag = tu_info.cotx_marker or ""
            if tu_info.aggregation_ratio_label:
                tag += f"\n{tu_info.aggregation_ratio_label}"

            anno_id = f"anno_{source_id}_{r_start}_{c_start}"
            anno = SourceAnnotation(
                id=anno_id,
                parent=self,
                marker=tu_info.cotx_marker,
                tag_label=tag.strip(),
                source_type="plasmid" if tu_info.in_l2 else "cotx",
                min_dimensions=Size(),
                max_dimensions=Size(),
                offset=Offset(),
                z_index=-5,
            )
            self._annotation_components[anno_id] = anno

        self._log_debug(
            f"built components: {len(self._tu_components)} TUs, {len(self._annotation_components)} Annotations, {len(self._connection_components)} Connections"
        )

    def _calculate_and_position_grid(self, measured_tu_sizes: Dict[str, Size]):
        """Calculates grid cell sizes and sets absolute offsets for TUs."""
        # (Mostly same logic as before, ensures parent link)
        rows = self._grid_dimensions.get("num_rows", 0)
        cols = self._grid_dimensions.get("num_cols", 0)
        if rows == 0 or cols == 0:
            self._grid_dimensions = {}
            return

        col_w = [0.0] * cols
        row_h = [0.0] * rows
        for tu_id, (r, c) in self._grid_coords.items():
            if tu_id in measured_tu_sizes:
                size = measured_tu_sizes[tu_id]
                col_w[c] = max(col_w[c], size.width)
                row_h[r] = max(row_h[r], size.height)

        col_gap, row_gap = self.grid_gap
        col_x = [sum(col_w[:c]) + c * col_gap for c in range(cols)]
        row_y = [sum(row_h[:r]) + r * row_gap for r in range(rows)]

        self._grid_dimensions.update(
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
                if tu_id not in measured_tu_sizes:
                    continue
                tu_size = measured_tu_sizes[tu_id]
                cell_w, cell_h = col_w[c], row_h[r]
                off_x = (cell_w - tu_size.width) / 2.0 if cell_w > tu_size.width else 0
                off_y = (cell_h - tu_size.height) / 2.0 if cell_h > tu_size.height else 0
                comp.offset = Offset(absolute=(col_x[c] + off_x, row_y[r] + off_y))
                comp.parent = self  # ensure link

        self._log_debug("calculated grid dimensions and set tu offsets.")

    def _calculate_and_position_annotations(self):
        """Calculates size and position for SourceAnnotations based on grid dimensions."""
        if not self._grid_dimensions or "col_x" not in self._grid_dimensions:
            return  # grid not calculated

        col_x, row_y = self._grid_dimensions["col_x"], self._grid_dimensions["row_y"]
        col_w, row_h = self._grid_dimensions["col_widths"], self._grid_dimensions["row_heights"]
        col_gap, row_gap = self.grid_gap

        for anno_id, anno in self._annotation_components.items():
            # find the corresponding source group coords
            source_id = "_".join(anno_id.split("_")[1:-2])  # extract source_id from anno_id
            group_coords = self._source_groups.get(source_id)
            if not group_coords:
                continue

            # find min/max row/col indices in the group
            min_r = min(r for r, c in group_coords)
            max_r = max(r for r, c in group_coords)
            min_c = min(c for r, c in group_coords)
            max_c = max(c for r, c in group_coords)

            # calculate bounding box based on grid cells
            local_min_x = col_x[min_c]
            local_min_y = row_y[min_r]
            local_max_x = col_x[max_c] + col_w[max_c]
            local_max_y = row_y[max_r] + row_h[max_r]

            jstyle.apply(anno)  # ensure padding is applied from styles
            pad_t, pad_r, pad_b, pad_l = anno.style.padding

            anno_w = (local_max_x - local_min_x) + pad_l + pad_r
            anno_h = (local_max_y - local_min_y) + pad_t + pad_b
            anno_x_local = local_min_x - pad_l
            anno_y_local = local_min_y - pad_t

            # update annotation size and position
            anno.min_dimensions = Size(width=anno_w, height=anno_h)
            anno.max_dimensions = Size(width=anno_w, height=anno_h)
            anno.offset = Offset(absolute=(anno_x_local, anno_y_local))
            anno.parent = self  # ensure link
            self._log_debug(
                f"positioned annotation {anno_id}: offset={anno.offset}, size={anno.min_dimensions}"
            )

    def measure_and_layout(self, renderer: Optional[BaseRenderer] = None) -> Size:
        """Measures and lays out the schematic components."""
        self._log_debug("measure_and_layout start")
        jstyle.apply(self)
        self.children = []  # clear children from previous runs
        self._content_offset = (0.0, 0.0)

        # Ensure components are built (should be done in post_init)
        if not self._component_factory or not self._tu_components:
            self._log_debug("components not built, attempting build now.")
            self._build_components()
            if not self._tu_components:
                self._log_debug("no tu components to layout after build.")
                return super().measure_and_layout(renderer)

        # --- Layout Pass ---
        # 1. Apply styles & Measure TUs
        measured_tu_sizes = {}
        for tu_id, comp in self._tu_components.items():
            jstyle.apply(comp)
            size_result = comp.measure_and_layout(renderer)
            if not isinstance(size_result, Size):
                size_result = Size()
            measured_tu_sizes[tu_id] = size_result

        # 2. Calculate Grid & Position TUs
        self._calculate_and_position_grid(measured_tu_sizes)
        if not self._grid_dimensions:
            return super().measure_and_layout(renderer)

        # 3. Calculate & Position Annotations (based on grid dimensions)
        self._calculate_and_position_annotations()

        # 4. Apply Styles & Measure Annotations
        for anno in self._annotation_components.values():
            jstyle.apply(anno)
            anno.measure_and_layout(renderer)  # measure annotation *after* its size/pos is set

        # 5. Apply Styles & Measure Connections (depends on final part positions)
        for conn in self._connection_components:
            jstyle.apply(conn)
            conn.measure_and_layout(renderer)

        # 6. Add all components to children list for rendering order and bounding box
        self.add_children(self._tu_components.values())
        self.add_children(self._annotation_components.values())
        self.add_children(self._connection_components)

        # --- Final Sizing and Shift ---
        # 7. Calculate content bounds based on final child positions/sizes
        min_x, min_y, max_x, max_y = self._calculate_content_bounds()

        # 8. Determine final schematic size including padding
        pad_t, pad_r, pad_b, pad_l = self.style.padding
        content_w = max(0, max_x - min_x)
        content_h = max(0, max_y - min_y)
        req_w = content_w + pad_l + pad_r
        req_h = content_h + pad_t + pad_b
        final_w = max(self.min_dimensions.width, req_w)
        final_h = max(self.min_dimensions.height, req_h)
        self._dimensions = Size(width=final_w, height=final_h)

        # 9. Calculate and apply content shift for padding
        shift_x = pad_l - min_x
        shift_y = pad_t - min_y
        self._content_offset = (shift_x, shift_y)
        if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
            self._log_debug(f"applying content shift: dx={shift_x:.2f}, dy={shift_y:.2f}")
            for child in self.children:
                if hasattr(child, "offset") and isinstance(child.offset, Offset):
                    orig_abs = child.offset.absolute
                    child.offset.absolute = (orig_abs[0] + shift_x, orig_abs[1] + shift_y)

        self._log_debug(f"final schematic layout complete, dimensions: {self._dimensions}")
        return self._dimensions

    def _calculate_content_bounds(self) -> Tuple[float, float, float, float]:
        """Calculates the min/max local coordinates occupied by all children."""
        if not self.children:
            return (0, 0, 0, 0)
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        found = False
        for c in self.children:
            if (
                c
                and c.show
                and hasattr(c, "_dimensions")
                and c._dimensions.width >= 0
                and c._dimensions.height >= 0
            ):
                ox, oy = c.offset.absolute
                w, h = c._dimensions.width, c._dimensions.height
                min_x = min(min_x, ox)
                min_y = min(min_y, oy)
                max_x = max(max_x, ox + w)
                max_y = max(max_y, oy + h)
                found = True
        return (min_x, min_y, max_x, max_y) if found else (0, 0, 0, 0)
