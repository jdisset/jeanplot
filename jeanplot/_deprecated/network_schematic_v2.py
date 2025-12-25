"""Network genetic schematic for GraphState-based networks"""

from typing import Any, Literal
from pydantic import Field, PrivateAttr
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
from jeanplot._deprecated.network_adapter import get_tu_informations_v2, get_interactions_v2, TUInfo
from jeanplot._deprecated.network_utils import Interaction
from jeanplot.core.style import jstyle
from jeanplot.core.renderer import BaseRenderer
from jeanplot.core.grid_utils import CellRegion


class SourceAnnotation(Container):
    is_overlay: bool = True
    source_id: str
    has_tag: bool = False
    source_type: Literal["plasmid", "cotx"] | None = "cotx"
    marker: str | None = None
    tag_label: str | None = None
    style_class: list[str] = ["source_annotation"]
    _source_proxy: Source | None = PrivateAttr(default=None)
    _cell_region: CellRegion | None = PrivateAttr(default=None)
    _cell_bounds: dict[tuple[int, int], tuple[float, float, float, float]] | None = PrivateAttr(
        default=None
    )
    _original_offset: tuple[float, float] | None = PrivateAttr(default=None)
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )
    style: BoxStyle = Field(default_factory=BoxStyle)

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        if self.has_tag:
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

    def render(self, renderer, context, matrix):
        if not self.show:
            return

        has_border = (
            self.style.border_width > 0
            and self.style.border_color
            and self.style.border_color.lower() != "none"
        )
        rendered_border = False

        if has_border and self._cell_region and self._cell_bounds:
            edges = self._cell_region.compute_boundary_edges(
                self._cell_bounds, self._original_offset or (0.0, 0.0)
            )
            if edges:
                border_style = self.style.model_copy(
                    update={"background_color": None, "shadow": None}
                )
                renderer.render_edges(context, edges, border_style, matrix, component=self)
                rendered_border = True

        if self.debug:
            renderer.render_debug(context, self, matrix)

        if rendered_border:
            saved = self.style.border_width
            self.style.border_width = 0
            Container.render(self, renderer, context, matrix)
            self.style.border_width = saved
        else:
            Container.render(self, renderer, context, matrix)


EL_MAP = {
    "ERN": (ERN, True),
    "ERN_recog_site_5p": (ERN5pRecog, True),
    "promoter": (Promoter, False),
    "terminator": (Terminator, False),
    "fluo_marker": (FluoMarker, True),
    "uORF_group": (UorfGroup, True),
}
CURVE_MAP = {"orthogonal": OrthogonalCurve, "bezier": SimpleBezierCurve, "straight": StraightCurve}


class SchematicComponentFactoryV2:
    def __init__(
        self, tu_infos: dict[str, TUInfo], interactions: list[Interaction], connection_style: str
    ):
        self._tu_infos, self._interactions = tu_infos, interactions
        self._connection_style = connection_style
        self._part_cache: dict[tuple[str, str], Component] = {}

    def _create_element(self, part_info: dict, tu_id: str) -> Component | None:
        cat, name = part_info.get("category", "unknown"), part_info.get("name", "")
        if cat == "insulator":
            return None
        el_class, needs_name = EL_MAP.get(cat, (GeneticPart, True))
        kwargs = {"id": f"{tu_id}_{name}"}
        if el_class is GeneticPart:
            kwargs["part_type"] = cat
        if needs_name:
            kwargs["part_name"] = name
        element = el_class(**kwargs)
        self._part_cache[(tu_id, name)] = element
        return element

    def create_tu(self, tu_id: str) -> TranscriptionUnit | None:
        if tu_id not in self._tu_infos:
            return None
        info = self._tu_infos[tu_id]
        tu = TranscriptionUnit(id=tu_id, name=info.tu_name)
        for part in info.parts:
            if el := self._create_element(part, tu_id):
                tu.add_child(el)
        return tu

    def create_connection(
        self, interaction: Interaction, index: int, parent: Component
    ) -> Connection | None:
        src = self._part_cache.get((interaction.src_tu_id, interaction.src_part_name))
        tgt = self._part_cache.get((interaction.tgt_tu_id, interaction.tgt_part_name))
        if not src or not tgt:
            return None
        return Connection(
            id=f"connection_{index}",
            parent=parent,
            style_class=[
                f"connection_{interaction.type}",
                f"connection_{interaction.type}_{interaction.src_part_name}",
            ],
            start_component=src,
            end_component=tgt,
            curve_type=CURVE_MAP.get(self._connection_style, StraightCurve)(),
            end_cap=LineEndFlat(),
            is_overlay=True,
            z_index=10,
            auto_route=True,
        )


def _get_source_id(tu_id: str, tu_infos: dict[str, TUInfo]) -> str:
    info = tu_infos.get(tu_id)
    if not info:
        return "unknown_source"
    return (
        f"plasmid_{info.plasmid_name}"
        if info.in_l2
        else f"source_{info.cotx_marker}_{info.aggregation_node_id}_{info.aggregation_ratio_label}"
    )


class NetworkGeneticSchematicV2(Container):
    network: Any
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(30, 30, 30, 30)))
    layout_orientation: Literal["row", "column"] = "column"
    hide_marker_tus: bool = True
    grid_gap: tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="start", justify_content="start")
    )

    _tu_infos: dict[str, TUInfo] = PrivateAttr(default_factory=dict)
    _interactions: list[Interaction] = PrivateAttr(default_factory=list)
    _grid_coords: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)
    _grid_dimensions: dict[str, Any] = PrivateAttr(default_factory=dict)
    _component_factory: SchematicComponentFactoryV2 | None = PrivateAttr(default=None)
    _tu_components: dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _annotation_components: list[SourceAnnotation] = PrivateAttr(default_factory=list)
    _connection_components: list[Connection] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any = None):
        super().model_post_init(__context)
        if self.network and self.network.compute_graph:
            self._process_network_data()
            self._build_components()

    def _process_network_data(self):
        self._tu_infos = get_tu_informations_v2(self.network)
        self._interactions = get_interactions_v2(self.network)
        graph = self.network.compute_graph

        # topological ordering
        tu_to_layer = {}
        for layer_idx, layer_nodes in enumerate(
            graph.topological_order([n.node_id for n in graph.get_nodes_by_type("translation")])
        ):
            for nid in layer_nodes:
                for edge in graph.get_incoming_edges(nid):
                    self._trace_to_source(edge.source_id, layer_idx, tu_to_layer)

        # visible TUs (excluding markers if hidden)
        visible = {
            t for t in self._tu_infos if not (self.hide_marker_tus and self._tu_infos[t].is_marker)
        }
        if not visible:
            self._grid_coords, self._grid_dimensions = {}, {"num_rows": 0, "num_cols": 0}
            return

        # group by source and column
        cotx_by_col: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for tu_id in visible:
            cotx_by_col[_get_source_id(tu_id, self._tu_infos)][tu_to_layer.get(tu_id, 0)].append(
                tu_id
            )

        cotx_by_col = {k: dict(v) for k, v in cotx_by_col.items()}
        num_cols = max((layer for tbc in cotx_by_col.values() for layer in tbc), default=0) + 1

        # sort cotx by size then earliest column
        sorted_cotx = sorted(
            cotx_by_col,
            key=lambda s: (
                -sum(len(v) for v in cotx_by_col[s].values()),
                min(cotx_by_col[s], default=0),
            ),
        )

        # strip-packing placement
        occupancy, positions, max_row = {}, {}, 0
        for sid in sorted_cotx:
            tbc = cotx_by_col[sid]
            if not tbc:
                continue
            cols = sorted(tbc)
            min_c, max_c = min(cols), max(cols)
            rows_needed = max(len(tbc[c]) for c in cols)

            start = 0
            while any(
                (r, c) in occupancy
                for r in range(start, start + rows_needed)
                for c in range(min_c, max_c + 1)
            ):
                start += 1

            for col in cols:
                for i, tu_id in enumerate(tbc[col]):
                    positions[tu_id] = (start + i, col)
                    occupancy[(start + i, col)] = sid

            for r in range(start, start + rows_needed):
                for c in range(min_c, max_c + 1):
                    occupancy.setdefault((r, c), sid)
            max_row = max(max_row, start + rows_needed)

        self._compact_grid(positions, num_cols, max_row)

    def _trace_to_source(
        self, node_id: int, layer_idx: int, tu_to_layer: dict, visited: set = None
    ):
        if visited is None:
            visited = set()
        if node_id in visited:
            return
        visited.add(node_id)

        graph = self.network.compute_graph
        node = graph.nodes.get(node_id)
        if not node:
            return

        if node.node_type == "source":
            name, cotx = node.extra.get("name", ""), node.extra.get("cotx_group", "cotx_1")
            tu_id = f"{name}_{cotx}" if name else f"tu_{node_id}"
            tu_to_layer.setdefault(tu_id, layer_idx)
        elif node.node_type == "sequestron_ERN":
            for e in graph.get_incoming_edges(node_id):
                if e.to_input_slot == 1:
                    self._trace_to_source(e.source_id, layer_idx, tu_to_layer, visited)
        else:
            for e in graph.get_incoming_edges(node_id):
                self._trace_to_source(e.source_id, layer_idx, tu_to_layer, visited)

    def _compact_grid(self, positions: dict[str, tuple[int, int]], raw_cols: int, raw_rows: int):
        self._grid_coords.clear()
        if not positions:
            self._grid_dimensions = {"num_rows": 0, "num_cols": 0}
            return

        rows = sorted({r for r, _ in positions.values()})
        cols = sorted({c for _, c in positions.values()})
        row_map, col_map = {o: n for n, o in enumerate(rows)}, {o: n for n, o in enumerate(cols)}

        for tu_id, (r, c) in positions.items():
            self._grid_coords[tu_id] = (row_map[r], col_map[c])
        self._grid_dimensions = {"num_rows": len(rows), "num_cols": len(cols)}

    def _build_components(self):
        if not self._tu_infos:
            return

        self._component_factory = SchematicComponentFactoryV2(
            self._tu_infos, self._interactions, self.connection_style
        )
        self._tu_components = {
            t: c for t in self._grid_coords if (c := self._component_factory.create_tu(t))
        }
        self._connection_components = [
            c
            for i, inter in enumerate(self._interactions)
            if (c := self._component_factory.create_connection(inter, i, self))
        ]

        # group cells and TUs by source
        source_cells: dict[str, list[tuple[int, int]]] = defaultdict(list)
        source_tus: dict[str, list[str]] = defaultdict(list)
        for tu_id, (r, c) in self._grid_coords.items():
            if tu_id in self._tu_infos:
                sid = _get_source_id(tu_id, self._tu_infos)
                source_cells[sid].append((r, c))
                source_tus[sid].append(tu_id)

        # create annotations
        self._annotation_components = []
        for sid, cells in source_cells.items():
            tus = source_tus[sid]
            if not tus:
                continue

            marker_tu = next(
                (
                    self._tu_infos[t]
                    for t in tus
                    if self._tu_infos.get(t, TUInfo(tu_id="", tu_name="")).is_marker
                ),
                None,
            )
            top_left = min(tus, key=lambda t: self._grid_coords.get(t, (999, 999)))
            tag_tu = self._tu_infos.get(top_left) or self._tu_infos.get(tus[0])
            if not tag_tu:
                continue

            marker = marker_tu.cotx_marker if marker_tu else tag_tu.cotx_marker
            tag_parts = [
                p for p in [tag_tu.cotx_name or marker, tag_tu.aggregation_ratio_label] if p
            ]

            anno = SourceAnnotation(
                id=f"anno_{sid}",
                parent=self,
                source_id=sid,
                has_tag=True,
                marker=marker,
                tag_label="\n".join(tag_parts),
                source_type="plasmid" if tag_tu.in_l2 else "cotx",
                z_index=-5,
            )
            anno._cell_region = CellRegion(cells=list(cells))
            self._annotation_components.append(anno)

    def _position_components(self, measured_sizes: dict[str, Size]):
        rows, cols = (
            self._grid_dimensions.get("num_rows", 0),
            self._grid_dimensions.get("num_cols", 0),
        )
        if not rows or not cols:
            self._grid_dimensions = {}
            return

        # compute dimensions
        col_w, row_h = [0.0] * cols, [0.0] * rows
        for tu_id, (r, c) in self._grid_coords.items():
            if sz := measured_sizes.get(tu_id):
                col_w[c], row_h[r] = max(col_w[c], sz.width), max(row_h[r], sz.height)

        # get annotation padding
        pad = margin = (0, 0, 0, 0)
        if self._annotation_components:
            sample = self._annotation_components[0]
            jstyle.apply(sample)
            pad, margin = sample.style.padding, sample.style.margin

        pt, pr, pb, pl = pad
        mt, mr, mb, ml = margin
        cell_w = [w + pl + pr for w in col_w]
        cell_h = [h + pt + pb for h in row_h]

        cgap, rgap = self.grid_gap
        col_x = [sum(cell_w[:c]) + c * cgap for c in range(cols)]
        row_y = [sum(cell_h[:r]) + r * rgap for r in range(rows)]

        self._grid_dimensions.update(
            {
                "cell_widths": cell_w,
                "cell_heights": cell_h,
                "col_x": col_x,
                "row_y": row_y,
                "total_width": sum(cell_w) + max(0, cols - 1) * cgap,
                "total_height": sum(cell_h) + max(0, rows - 1) * rgap,
            }
        )

        # position TUs
        for tu_id, (r, c) in self._grid_coords.items():
            if (comp := self._tu_components.get(tu_id)) and (sz := measured_sizes.get(tu_id)):
                comp.offset = Offset(
                    absolute=(
                        col_x[c] + (cell_w[c] - sz.width) / 2,
                        row_y[r] + (cell_h[r] - sz.height) / 2,
                    )
                )
                comp.parent = self

        # position annotations
        for anno in self._annotation_components:
            region = anno._cell_region
            if not region or not region.cells:
                continue

            cell_bounds = {
                (r, c): (
                    col_x[c] + ml,
                    row_y[r] + mt,
                    col_x[c] + cell_w[c] - mr,
                    row_y[r] + cell_h[r] - mb,
                )
                for (r, c) in region.cells
                if c < cols and r < rows
            }
            anno._cell_bounds = cell_bounds

            rmin, rmax, cmin, cmax = region.bounds()
            ax, ay = col_x[cmin] + ml, row_y[rmin] + mt
            aw = col_x[cmax] + cell_w[cmax] - col_x[cmin] - ml - mr
            ah = row_y[rmax] + cell_h[rmax] - row_y[rmin] - mt - mb

            jstyle.apply(anno)
            anno.min_dimensions = anno.max_dimensions = Size(width=aw, height=ah)
            anno.offset = Offset(absolute=(ax, ay))
            anno._original_offset = (ax, ay)
            anno.parent = self

    def measure_and_layout(self, renderer: BaseRenderer | None = None) -> Size:
        jstyle.apply(self)
        self.children = []

        if not self._component_factory or not self._tu_components:
            self._dimensions = self._apply_constraints(Size())
            return self._dimensions

        measured = {}
        for tu_id, comp in self._tu_components.items():
            jstyle.apply(comp)
            sz = comp.measure_and_layout(renderer)
            measured[tu_id] = sz if isinstance(sz, Size) else Size()

        self._position_components(measured)
        if not self._grid_dimensions:
            self._dimensions = self._apply_constraints(Size())
            return self._dimensions

        for c in self._annotation_components + self._connection_components:
            jstyle.apply(c)
            c.measure_and_layout(renderer)

        self.add_children(self._annotation_components)
        self.add_children(self._tu_components.values())
        self.add_children(self._connection_components)

        # compute bounds
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for c in self.children:
            if (
                c
                and c.show
                and hasattr(c, "_dimensions")
                and isinstance(getattr(c, "offset", None), Offset)
                and c._dimensions.width >= 0
            ):
                ox, oy = c.offset.absolute
                min_x, min_y = min(min_x, ox), min(min_y, oy)
                max_x, max_y = (
                    max(max_x, ox + c._dimensions.width),
                    max(max_y, oy + c._dimensions.height),
                )

        if min_x == float("inf"):
            min_x = min_y = max_x = max_y = 0

        pt, pr, pb, pl = self.style.padding
        cw, ch = max(0, max_x - min_x), max(0, max_y - min_y)

        self._natural_dimensions = Size(width=cw + pl + pr, height=ch + pt + pb)
        self._dimensions = self._apply_constraints(self._natural_dimensions)

        shift_x, shift_y = pl - min_x, pt - min_y
        if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
            for child in self.children:
                if isinstance(getattr(child, "offset", None), Offset):
                    ox, oy = child.offset.absolute
                    child.offset.absolute = (ox + shift_x, oy + shift_y)

        return self._dimensions

    def _layout_children(self, renderer: BaseRenderer | None):
        for child in self.children:
            if isinstance(child, Container):
                child._layout_children(renderer)
