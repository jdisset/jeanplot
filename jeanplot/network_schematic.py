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
from pydantic import Field, PrivateAttr
from collections import defaultdict
import numpy as np

from .component import Component
from .container import Container
from .models import Size, BoxStyle, LayoutConstraints, Offset, Transform
from .genetic_elements import (
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
from .connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from .svg import LineEndFlat
from .network_utils import (
    get_tu_informations,
    get_tu_grid_layout,
    get_interactions,
    optimize_grid_for_source_adjacency,
    _get_source_id,  # reuse helper
)
from .network_utils import TUInfo, Interaction
from .style import jstyle


class SourceAnnotation(Container):
    is_overlay: bool = True
    source_type: Optional[Literal["plasmid", "cotx"]] = "cotx"
    marker: Optional[str] = None
    tag_label: Optional[str] = None
    style_class: list[str] = ["source_annotation"]
    _source_style_proxy: Source = PrivateAttr(default=None)
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        proxy = Source(
            source_type=self.source_type, marker=self.marker, tag_label=self.tag_label, children=[]
        )
        jstyle.apply(proxy)
        self.style = proxy.style.model_copy()
        if proxy.tag_content:
            self.children = [proxy.tag_content]
            proxy.tag_content.parent = self

    def measure(self, renderer=None) -> Size:
        self._dimensions.width = max(0.0, getattr(self._dimensions, "width", 0.0))
        self._dimensions.height = max(0.0, getattr(self._dimensions, "height", 0.0))
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def measure_and_layout(self, renderer=None) -> Size:
        jstyle.apply(self)
        self.measure(renderer)
        Container.apply_layout(self)
        for child in self.children:
            child.measure_and_layout(renderer)
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        if self.style.background_color or (self.style.border_color and self.style.border_width > 0):
            renderer.render_rectangle(context, self._dimensions, self.style, matrix, component=self)
        if self.debug:
            renderer.render_debug(context, self, matrix)
        Container.render(self, renderer, context, matrix)


class NetworkGeneticSchematic(Container):
    network: Any
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="center", justify_content="start")
    )
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(background_color="#ffffff", padding=(30, 30, 30, 30))
    )
    layout_orientation: Literal["row", "column"] = "row"
    show_all_tus: bool = False
    grid_gap: Tuple[float, float] = (40.0, 40.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"

    _tu_infos: Dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    # _raw_grid_layout no longer needed, filter happens before transposition
    _compacted_grid_layout: List[List[Optional[str]]] = PrivateAttr(
        default_factory=list
    )  # row-major, filtered and compacted
    _interactions: List[Interaction] = PrivateAttr(default_factory=list)
    _tu_components: Dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _part_components: Dict[Tuple[str, str], Component] = PrivateAttr(default_factory=dict)
    _source_map: Dict[Tuple[int, int], str] = PrivateAttr(
        default_factory=dict
    )  # uses compacted coords (r,c) -> source_id
    _grid_coords: Dict[str, Tuple[int, int]] = PrivateAttr(
        default_factory=dict
    )  # tu_id -> compacted coords (r,c)
    _calculated_grid: Dict[str, Any] = PrivateAttr(
        default_factory=dict
    )  # stores compacted dims, col_widths, row_heights etc.

    def model_post_init(self, *args, **kwargs):
        self._tu_infos = get_tu_informations(self.network)
        raw_grid_layers = get_tu_grid_layout(self.network)
        optimized_layers = optimize_grid_for_source_adjacency(raw_grid_layers, self._tu_infos)
        self._interactions = get_interactions(self.network)

        # filter markers *before* preparing/compacting grid
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

        # prepare row-major layout (transposes if needed) using filtered layers
        transposed_layout = self._transpose_if_needed(filtered_layers)
        # compact layout (remove empty rows/cols) and build maps
        self._compact_layout_and_build_maps(transposed_layout)
        # create TUs based on _grid_coords (which uses compacted indices)
        self._create_tu_components(add_as_children=False)

    def _transpose_if_needed(self, layers: List[List[str]]) -> List[List[Optional[str]]]:
        """transposes layers if orientation is 'column', ensures uniform width"""
        if self.layout_orientation == "column":
            max_inner_len = max((len(sublist) for sublist in layers), default=0)
            # pad shorter layers with None before transposing
            padded_layers = [
                list(layer) + [None] * (max_inner_len - len(layer)) for layer in layers
            ]
            # transpose
            grid = [
                [padded_layers[j][i] for j in range(len(padded_layers))]
                for i in range(max_inner_len)
            ]
        else:
            grid = [[cell for cell in row] for row in layers]  # mutable copy

        # ensure uniform width even in row mode
        if grid:
            max_len = max((len(row) for row in grid), default=0)
            for row in grid:
                row.extend([None] * (max_len - len(row)))
        return grid

    def _compact_layout_and_build_maps(self, layout: List[List[Optional[str]]]):
        """removes empty rows/columns from layout -> _compacted_grid_layout and builds maps"""
        self._compacted_grid_layout = layout  # start with the input layout
        if not self._compacted_grid_layout:
            self._source_map.clear()
            self._grid_coords.clear()
            self._calculated_grid.update({"num_rows": 0, "num_cols": 0})
            return

        raw_rows = len(self._compacted_grid_layout)
        raw_cols = len(self._compacted_grid_layout[0])  # assumes uniform length

        non_empty_rows = {r for r in range(raw_rows) if any(self._compacted_grid_layout[r])}
        non_empty_cols = {
            c
            for c in range(raw_cols)
            if any(self._compacted_grid_layout[r][c] for r in range(raw_rows))
        }

        # handle case where filtering removed everything
        if not non_empty_rows or not non_empty_cols:
            self._compacted_grid_layout = []
            self._source_map.clear()
            self._grid_coords.clear()
            self._calculated_grid.update({"num_rows": 0, "num_cols": 0})
            return

        new_num_rows = len(non_empty_rows)
        new_num_cols = len(non_empty_cols)

        row_map = {old_r: new_r for new_r, old_r in enumerate(sorted(list(non_empty_rows)))}
        col_map = {old_c: new_c for new_c, old_c in enumerate(sorted(list(non_empty_cols)))}

        self._compacted_grid_layout = [[None] * new_num_cols for _ in range(new_num_rows)]
        self._source_map.clear()
        self._grid_coords.clear()

        for r_old in sorted(list(non_empty_rows)):
            for c_old in sorted(list(non_empty_cols)):
                tu_id = layout[r_old][c_old]  # use original layout before overwrite
                if tu_id:  # only process cells that originally had content
                    r_new, c_new = row_map[r_old], col_map[c_old]
                    self._compacted_grid_layout[r_new][c_new] = tu_id
                    self._grid_coords[tu_id] = (r_new, c_new)
                    s_id = _get_source_id(tu_id, self._tu_infos)
                    self._source_map[(r_new, c_new)] = s_id

        self._calculated_grid.update({"num_rows": new_num_rows, "num_cols": new_num_cols})

    def _create_tu_components(self, add_as_children=True):
        # uses compacted _grid_coords
        self._tu_components.clear()
        self._part_components.clear()
        if add_as_children:
            self.children = [c for c in self.children if not isinstance(c, TranscriptionUnit)]
        for tu_id in self._grid_coords.keys():  # iterate over TUs present in the compacted grid
            tu_info = self._tu_infos[tu_id]
            # create TU with name from tu_info
            tu = TranscriptionUnit(id=tu_id, name=tu_info.tu_name)
            tu.parent = self
            for part in tu_info.parts:
                element = self._create_genetic_element(part, tu_id)
                if element:
                    tu.add_child(element)
                    self._part_components[(tu_id, part.name)] = element
            self._tu_components[tu_id] = tu
            if add_as_children:
                self.children.append(tu)

    def _create_genetic_element(self, part_info: Interaction, tu_id: str) -> Optional[Component]:
        # same as before
        cat, name = part_info.category, part_info.name
        comp_id = f"{tu_id}_{name}"
        el_map = {
            "ERN": (ERN, True),
            "ERN_recog_site_5p": (ERN5pRecog, True),
            "promoter": (Promoter, True),
            "terminator": (Terminator, True),
            "fluo_marker": (FluoMarker, True),
            "uORF_group": (UorfGroup, True),
        }
        if cat == "insulator":
            return None
        el_class, needs_name = el_map.get(cat, (GeneticPart, True))
        kwargs = {"id": comp_id, "part_type": cat} if el_class is GeneticPart else {"id": comp_id}
        if needs_name:
            print(f"Creating {cat} component with name: {name}")
            kwargs["part_name"] = name
        return el_class(**kwargs)

    def _calculate_grid_layout(self, renderer=None):
        # uses compacted dimensions and _compacted_grid_layout
        rows = self._calculated_grid.get("num_rows", 0)
        cols = self._calculated_grid.get("num_cols", 0)
        if rows == 0 or cols == 0:
            self._calculated_grid.update(
                {
                    "col_widths": [],
                    "row_heights": [],
                    "col_x": [],
                    "row_y": [],
                    "total_width": 0,
                    "total_height": 0,
                    "measured_sizes": {},
                }
            )
            return

        measured = {
            tu_id: comp.measure_and_layout(renderer) for tu_id, comp in self._tu_components.items()
        }
        col_w = [
            max(
                (
                    measured[self._compacted_grid_layout[r][c]].width
                    for r in range(rows)
                    if self._compacted_grid_layout[r][c]
                ),
                default=0,
            )
            for c in range(cols)
        ]
        row_h = [
            max(
                (
                    measured[self._compacted_grid_layout[r][c]].height
                    for c in range(cols)
                    if self._compacted_grid_layout[r][c]
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
                "measured_sizes": measured,
            }
        )
        for tu_id, comp in self._tu_components.items():
            if tu_id in self._grid_coords:
                r, c = self._grid_coords[tu_id]  # compacted coords
                off_x = (col_w[c] - measured[tu_id].width) / 2
                off_y = (row_h[r] - measured[tu_id].height) / 2
                comp.transform.translate = (col_x[c] + off_x, row_y[r] + off_y)

    def _get_tu_bounds_in_world(
        self, r: int, c: int
    ) -> Optional[Tuple[float, float, float, float]]:
        # uses compacted grid indices
        tu_id = self._compacted_grid_layout[r][c]
        comp = self._tu_components.get(tu_id)
        if not comp:
            return None
        mat = comp.compute_local_matrix()
        w, h = comp._dimensions.width, comp._dimensions.height
        cn = (mat @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]])).T
        return (np.min(cn[:, 0]), np.min(cn[:, 1]), np.max(cn[:, 0]), np.max(cn[:, 1]))

    def _create_source_annotations(self):
        # uses compacted grid coordinates via _source_map
        rows = self._calculated_grid.get("num_rows", 0)
        cols = self._calculated_grid.get("num_cols", 0)
        if rows == 0 or cols == 0 or "col_widths" not in self._calculated_grid:
            return

        annotations_to_add = []
        covered_coords = set()
        for r_start in range(rows):
            for c_start in range(cols):
                start_coord = (r_start, c_start)
                if start_coord in covered_coords or start_coord not in self._source_map:
                    continue
                source_id = self._source_map[start_coord]

                # find largest possible rectangle starting here
                max_width = sum(
                    1
                    for c in range(c_start, cols)
                    if self._source_map.get((r_start, c)) == source_id
                )
                max_height = 0
                for r_test in range(r_start, rows):
                    if all(
                        self._source_map.get((r_test, c)) == source_id
                        for c in range(c_start, c_start + max_width)
                    ):
                        max_height = r_test - r_start + 1
                    else:
                        break
                if max_height == 0:
                    continue  # should not happen if start_coord is valid

                r_max, c_max = r_start + max_height - 1, c_start + max_width - 1
                rect_coords = [
                    (r, c) for r in range(r_start, r_max + 1) for c in range(c_start, c_max + 1)
                ]

                # calculate bounding box for this rectangular group
                all_bounds = [self._get_tu_bounds_in_world(r, c) for r, c in rect_coords]
                valid_bounds = [b for b in all_bounds if b]
                if not valid_bounds:
                    continue

                tu_min_x, tu_min_y = (
                    min(b[0] for b in valid_bounds),
                    min(b[1] for b in valid_bounds),
                )
                tu_max_x, tu_max_y = (
                    max(b[2] for b in valid_bounds),
                    max(b[3] for b in valid_bounds),
                )

                tu_info = self._tu_infos[self._compacted_grid_layout[r_start][c_start]]
                tag = f"{tu_info.cotx_marker}" + (
                    f" {tu_info.aggregation_ratio_label}" if tu_info.aggregation_ratio_label else ""
                )
                anno_id = f"anno_{source_id}_{r_start}_{c_start}"
                anno = SourceAnnotation(
                    id=anno_id,
                    parent=self,
                    marker=tu_info.cotx_marker,
                    tag_label=tag,
                    source_type="plasmid" if tu_info.in_l2 else "cotx",
                )
                pad_t, pad_r, pad_b, pad_l = anno.style.padding
                overlay_w, overlay_h = (
                    (tu_max_x - tu_min_x) + pad_l + pad_r,
                    (tu_max_y - tu_min_y) + pad_t + pad_b,
                )
                overlay_x, overlay_y = tu_min_x - pad_l, tu_min_y - pad_t
                anno.transform.translate = (overlay_x, overlay_y)
                anno._dimensions = Size(width=overlay_w, height=overlay_h)
                annotations_to_add.append(anno)
                covered_coords.update(rect_coords)
        self.children.extend(annotations_to_add)

    def _add_connections(self):
        # same as before
        connections_to_add = []
        curve_map = {
            "orthogonal": OrthogonalCurve,
            "bezier": SimpleBezierCurve,
            "straight": StraightCurve,
        }
        curve_type_cls = curve_map.get(self.connection_style, StraightCurve)
        for idx, interaction in enumerate(self._interactions):
            src = self._part_components.get((interaction.src_tu_id, interaction.src_part_name))
            tgt = self._part_components.get((interaction.tgt_tu_id, interaction.tgt_part_name))
            if not src or not tgt:
                continue
            conn = Connection(
                id=f"connection_{idx}",
                parent=self,
                style_class=[f"connection_{interaction.type}"],
                start_component=src,
                end_component=tgt,
                curve_type=curve_type_cls(),
                end_cap=LineEndFlat(length=8),
            )
            connections_to_add.append(conn)
        self.children.extend(connections_to_add)

    def measure_and_layout(self, renderer=None) -> Size:
        jstyle.apply(self)
        self.children = []
        self._create_tu_components(add_as_children=False)
        if not self._tu_components:
            self._dimensions = Size(
                width=sum(self.style.padding[1::2]), height=sum(self.style.padding[0::2])
            )
            self._transformed_aabb = self.compute_transformed_aabb()
            return self._dimensions
        self._calculate_grid_layout(renderer)
        self.children.extend(self._tu_components.values())
        self._create_source_annotations()
        self._add_connections()
        for child in self.children:
            if isinstance(child, (SourceAnnotation, Connection)):
                child.measure_and_layout(renderer)
            elif isinstance(child, TranscriptionUnit):
                child._transformed_aabb = child.compute_transformed_aabb()
        all_bounds = [c.get_local_bounds() for c in self.children]
        valid_bounds = [b for b in all_bounds if b]
        if not valid_bounds:
            min_x, min_y, max_x, max_y = 0, 0, 0, 0
        else:
            min_x, min_y, max_x, max_y = (
                min(b[0] for b in valid_bounds),
                min(b[1] for b in valid_bounds),
                max(b[2] for b in valid_bounds),
                max(b[3] for b in valid_bounds),
            )
        content_w, content_h = max_x - min_x, max_y - min_y
        pad = self.style.padding
        self._dimensions = Size(
            width=content_w + pad[1] + pad[3], height=content_h + pad[0] + pad[2]
        )
        off_x, off_y = pad[3] - min_x, pad[0] - min_y
        if abs(off_x) > 1e-6 or abs(off_y) > 1e-6:
            for child in self.children:
                tx, ty = child.transform.translate
                child.transform.translate = (tx + off_x, ty + off_y)
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions


def draw_network_schematic(
    network,
    figsize=(12, 8),
    dpi=200,
    show_all_tus=False,
    grid_gap=(40.0, 20.0),
    connection_style="orthogonal",
    layout_orientation: Literal["row", "column"] = "row",
    debug=False,
):
    schematic = NetworkGeneticSchematic(
        network=network,
        show_all_tus=show_all_tus,
        grid_gap=grid_gap,
        connection_style=connection_style,
        layout_orientation=layout_orientation,
        debug=debug,
    )
    from .matplotlib_renderer import MatplotlibRenderer
    import matplotlib.pyplot as plt

    renderer = MatplotlibRenderer(debug=debug)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    renderer.render_component(ax, schematic, adjust_lims=True)
    return fig, ax, schematic
