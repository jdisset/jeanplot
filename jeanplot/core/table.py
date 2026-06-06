from typing import Literal, Any, TypeVar
from pydantic import Field, model_validator, PrivateAttr, BaseModel
import numpy as np

from jeanplot.core.component import Component
from jeanplot.core.container import Container
from jeanplot.core.models import (
    BoxInset,
    Size,
    BoxStyle,
    LayoutConstraints,
)
from jeanplot.core.svg import SVGPathData
from jeanplot.core.text import Text
from jeanplot.core.style import jstyle

T = TypeVar("T")
ListOrSingle = T | list[T]


class CellStyle(BoxStyle):
    border_top: bool | None = None
    border_right: bool | None = None
    border_bottom: bool | None = None
    border_left: bool | None = None

    min_width: float | None = None
    max_width: float | None = None
    min_height: float | None = None
    max_height: float | None = None

    align_items: Literal["start", "center", "end", "stretch"] | None = None
    justify_content: (
        Literal["start", "center", "end", "space-between", "space-around", "space-evenly"] | None
    ) = None


class ColumnStyle(BaseModel):
    width: float | Literal["auto"] | str = "auto"
    cell_style: CellStyle | None = Field(default_factory=CellStyle)


class LineStyle(BaseModel):
    """One grid line's appearance. `color=None` or `width<=0` => not drawn."""

    color: str | None = None
    width: float = 0.0
    style: Literal["solid", "dashed", "dotted"] = "solid"
    dash_sequence: list[float] | None = None

    @property
    def visible(self) -> bool:
        return bool(self.color) and self.width > 0


class GridStyle(BaseModel):
    """SSOT for a Table's collapsed border grid. Each logical line type is described
    ONCE and the renderer draws every grid line a single time — no per-cell edge drawing,
    so shared edges are never doubled and the outer frame never overlaps an interior line.
    `frame` = the four-sided outer boundary; `header` = the separator below the header
    rows; `inner` = every interior row/column separator. `corner_radius` (inches) rounds
    the frame; interior lines meet the straight part of the frame, so they need no clip."""

    frame: LineStyle = Field(default_factory=LineStyle)
    header: LineStyle = Field(default_factory=LineStyle)
    inner: LineStyle = Field(default_factory=LineStyle)
    corner_radius: float = 0.0
    header_fill: str | None = None


class TableCell(Container):
    style: CellStyle = Field(default_factory=CellStyle)
    colspan: int = 1

    _row_index: int = PrivateAttr(default=0)
    _col_index: int = PrivateAttr(default=0)
    _is_header: bool = PrivateAttr(default=False)
    _styling: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def apply_styles(self):
        # jstyle.apply mutates fields; under validate_assignment every setattr re-fires
        # this validator. Guard so a direct cell-field set (debug, whole `style`) applies
        # the cascade once instead of recursing. (style stays CellStyle, not BoxStyle.)
        if self._styling:
            return self
        self._styling = True
        try:
            jstyle.apply(self)
            if not isinstance(self.style, CellStyle):
                self.style = CellStyle(**self.style.model_dump())
        finally:
            self._styling = False
        return self

    def measure_and_layout(self, renderer=None) -> Size:
        target_width = -1.0
        if self.style.align_items:
            self.layout.align_items = self.style.align_items
        if self.style.justify_content:
            self.layout.justify_content = self.style.justify_content

        if self.parent and hasattr(self.parent, "_calculated_column_widths"):
            col_widths = self.parent._calculated_column_widths
            if self._col_index < len(col_widths):
                effective_colspan = min(self.colspan, len(col_widths) - self._col_index)
                if effective_colspan > 0:
                    target_width = sum(
                        col_widths[c]
                        for c in range(self._col_index, self._col_index + effective_colspan)
                    )
                    # add inter-column gaps if borders are separate
                    if (
                        effective_colspan > 1
                        and hasattr(self.parent.parent, "border_collapse")
                        and self.parent.parent.border_collapse == "separate"
                        and hasattr(self.parent.parent, "border_spacing")
                    ):
                        target_width += self.parent.parent.border_spacing * (effective_colspan - 1)

                    self.min_dimensions.width = target_width
                    self.max_dimensions.width = target_width

        # cell style dims applied AFTER column target so they win
        if self.style.min_width is not None:
            self.min_dimensions.width = max(self.min_dimensions.width, self.style.min_width)
        if self.style.max_width is not None:
            self.max_dimensions.width = min(self.max_dimensions.width, self.style.max_width)
        if target_width >= 0 and self.min_dimensions.width > self.max_dimensions.width:
            self._log_debug(
                f"Warning: Cell style min/max width conflicts for cell {self._row_index},{self._col_index}. Using style constraints."
            )
            target_width = -1.0

        if self.style.min_height is not None:
            self.min_dimensions.height = max(self.min_dimensions.height, self.style.min_height)
        if self.style.max_height is not None:
            self.max_dimensions.height = min(self.max_dimensions.height, self.style.max_height)

        super().measure_and_layout(renderer)

        # enforce final width if column constraint was applied
        if target_width >= 0 and self.min_dimensions.width == self.max_dimensions.width:
            if abs(self._dimensions.width - target_width) > 1e-6:
                self._dimensions.width = target_width
        elif (
            self.max_dimensions.width < float("inf")
            and self._dimensions.width > self.max_dimensions.width
        ):
            self._dimensions.width = self.max_dimensions.width

        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render cell background/border then content."""
        effective_style = self.style
        draw_top = effective_style.border_top is not False
        draw_right = effective_style.border_right is not False
        draw_bottom = effective_style.border_bottom is not False
        draw_left = effective_style.border_left is not False

        if effective_style.background_color or (
            effective_style.border_color and effective_style.border_width > 0
        ):
            render_style = effective_style.model_copy()

            if not (draw_top and draw_right and draw_bottom and draw_left):
                # any side hidden: draw bg then individual border edges
                bg_style = render_style.model_copy()
                bg_style.border_width = 0
                bg_style.border_color = None
                if bg_style.background_color:
                    renderer.render_rectangle(
                        context, self._dimensions, bg_style, matrix, component=self
                    )

                if render_style.border_color and render_style.border_width > 0:
                    border_path = self._build_border_edges_path(
                        draw_top=draw_top,
                        draw_right=draw_right,
                        draw_bottom=draw_bottom,
                        draw_left=draw_left,
                    )
                    if border_path:
                        renderer.render_path(
                            context,
                            SVGPathData(
                                d=border_path,
                                fill=None,
                                stroke=render_style.border_color,
                                stroke_width=render_style.border_width,
                                line_style=render_style.border_style,
                                dash_array=render_style.dash_sequence,
                                dash_offset=render_style.dash_offset,
                            ),
                            matrix,
                            line_width_mode=render_style.border_width_mode,
                        )
            else:
                renderer.render_rectangle(
                    context, self._dimensions, render_style, matrix, component=self
                )

        if self.debug:
            renderer.render_debug(context, self, matrix)

        self._render_children(renderer, context, matrix)

    def _build_border_edges_path(
        self, *, draw_top: bool, draw_right: bool, draw_bottom: bool, draw_left: bool
    ) -> str:
        w = self._dimensions.width
        h = self._dimensions.height
        if w <= 0 or h <= 0:
            return ""

        segments: list[str] = []
        if draw_top:
            segments.append(f"M 0 0 L {w:.3f} 0")
        if draw_right:
            segments.append(f"M {w:.3f} 0 L {w:.3f} {h:.3f}")
        if draw_bottom:
            segments.append(f"M 0 {h:.3f} L {w:.3f} {h:.3f}")
        if draw_left:
            segments.append(f"M 0 0 L 0 {h:.3f}")

        return " ".join(segments)


class TableRow(Container):
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row",
            align_items="stretch",
            justify_content="start",
            gap=0,
        )
    )
    _row_index: int = PrivateAttr(default=0)
    _is_header: bool = PrivateAttr(default=False)
    _calculated_column_widths: list[float] = PrivateAttr(default_factory=list)


class Table(Container):
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column",
            align_items="stretch",
            justify_content="start",
            gap=0,
        )
    )
    style: BoxStyle = Field(default_factory=BoxStyle)
    # Collapsed-grid SSOT: the Table's frame + separators are described here and drawn
    # once by `_figure_render._draw_table_grid`. Cells/rows no longer carry grid borders
    # (their `style` is for backgrounds/padding/content only).
    grid: GridStyle = Field(default_factory=GridStyle)

    data: list[list[Any]] = Field(default_factory=list)
    column_styles: list[ColumnStyle] = Field(default_factory=list)
    header_rows: int = 0

    border_collapse: Literal["collapse", "separate"] = "collapse"
    border_spacing: float = 0

    _num_columns: int = PrivateAttr(default=0)
    _built_signature: tuple | None = PrivateAttr(default=None)

    def _build_signature(self) -> tuple:
        # Identity of the inputs that define the row/cell structure. `data` and
        # `column_styles` are validated as lists of model instances, so reassigning
        # either yields a fresh object (new id); in-place growth is caught by len.
        return (id(self.data), len(self.data), id(self.column_styles), self.header_rows)

    @model_validator(mode="after")
    def build_table(self):
        # Idempotent: `validate_assignment=True` re-fires every model_validator on any
        # field assignment, but rebuilding is destructive (it discards the TableRows
        # that hold computed column widths). Skip when the structural inputs are
        # unchanged, so the built table survives a Figure's measure/layout cycle.
        if self._built_signature == self._build_signature() and self.children:
            return self
        # clear in place to avoid assignment-validation recursion
        self.children.clear()
        if not self.data:
            self._num_columns = 0
            self._built_signature = self._build_signature()
            return self

        max_effective_cols = 0
        for row_data in self.data:
            current_effective_cols = 0
            for cell_content in row_data:
                colspan = 1
                if isinstance(cell_content, TableCell):
                    colspan = cell_content.colspan
                elif isinstance(cell_content, dict) and "colspan" in cell_content:
                    colspan = cell_content.get("colspan", 1)

                current_effective_cols += colspan
            max_effective_cols = max(max_effective_cols, current_effective_cols)
        self._num_columns = max_effective_cols

        if len(self.column_styles) < self._num_columns:
            self.column_styles.extend(
                [ColumnStyle() for _ in range(self._num_columns - len(self.column_styles))]
            )

        for r, row_data in enumerate(self.data):
            is_header = r < self.header_rows
            table_row = TableRow(_row_index=r, _is_header=is_header)
            table_row.style_class += ["table-row", f"row-{r}"]
            if is_header:
                table_row.style_class.append("table-header-row")
            if r == 0:
                table_row.style_class.append("row-first")
            if r == len(self.data) - 1:
                table_row.style_class.append("row-last")
            table_row.style_class.append("row-even" if r % 2 == 0 else "row-odd")

            col_offset = 0
            for c_nominal, cell_content in enumerate(row_data):
                c_actual = col_offset
                if c_actual >= self._num_columns:
                    self._log_debug(
                        f"Warning: Cell data in row {r} exceeds calculated column count ({self._num_columns}) due to colspan. Ignoring extra cell."
                    )
                    break

                if isinstance(cell_content, TableCell):
                    cell = cell_content
                    cell.colspan = min(cell.colspan, self._num_columns - c_actual)
                elif isinstance(cell_content, Component):
                    cell = TableCell(children=[cell_content])
                    cell.colspan = min(cell.colspan, self._num_columns - c_actual)
                else:
                    cell = TableCell(children=[Text(text=str(cell_content))])
                    cell.colspan = min(cell.colspan, self._num_columns - c_actual)

                cell._row_index = r
                cell._col_index = c_actual
                cell._is_header = is_header

                cell.style_class += [
                    "table-cell",
                    f"col-{c_actual}",
                    f"cell-{r}-{c_actual}",
                    f"row-{r}",
                    "row-even" if r % 2 == 0 else "row-odd",
                    "col-even" if c_actual % 2 == 0 else "col-odd",
                ]
                if is_header:
                    cell.style_class.append("table-header-cell")
                if c_actual == 0:
                    cell.style_class.append("col-first")
                if c_actual + cell.colspan >= self._num_columns:
                    cell.style_class.append("col-last")
                if r == 0:
                    cell.style_class.append("row-first")
                if r == len(self.data) - 1:
                    cell.style_class.append("row-last")

                base_style = (
                    self.column_styles[c_actual].cell_style.model_copy()
                    if c_actual < len(self.column_styles)
                    else CellStyle()
                )

                # exclude_unset: only carry fields actually set on the column default or
                # the cell, so the rest stay fillable by the jstyle cascade. A full dump
                # would mark every field user-set and the fill strategy would skip them.
                merged_style_dict = self._merge_styles(
                    base_style.model_dump(exclude_unset=True),
                    cell.style.model_dump(exclude_unset=True),
                )
                cell.style = CellStyle(**merged_style_dict)

                table_row.add_child(cell)

                col_offset += cell.colspan

            if self.border_collapse == "separate" and self.border_spacing > 0:
                table_row.layout.gap = self.border_spacing
                table_row.style.padding = BoxInset(
                    top=self.border_spacing / 2, bottom=self.border_spacing / 2
                )

            self.add_child(table_row)

        if self.border_collapse == "separate" and self.border_spacing > 0:
            pad = self.style.padding
            bump = self.border_spacing / 2
            self.style.padding = BoxInset(
                top=pad.top + bump,
                right=pad.right + bump,
                bottom=pad.bottom + bump,
                left=pad.left + bump,
            )
            self.layout.gap = self.border_spacing

        self._built_signature = self._build_signature()
        return self

    def _merge_styles(self, base: dict, overlay: dict) -> dict:
        """merge two style dicts; overlay wins for non-None values."""
        merged = base.copy()
        for key, value in overlay.items():
            if value is not None:
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_styles(merged[key], value)
                else:
                    merged[key] = value
        return merged

    def measure_and_layout(self, renderer=None) -> Size:
        """overrides container layout to compute column widths."""
        if not self.children:
            return super().measure_and_layout(renderer)

        # first pass: measure natural cell sizes with constraints relaxed
        original_min_max = {}
        for r, row in enumerate(self.children):
            if isinstance(row, TableRow):
                for c, cell in enumerate(row.children):
                    if isinstance(cell, TableCell):
                        key = (r, c)
                        original_min_max[key] = (
                            cell.min_dimensions.model_copy(),
                            cell.max_dimensions.model_copy(),
                        )
                        cell.min_dimensions.width = 0
                        cell.max_dimensions.width = float("inf")

        super().measure_and_layout(renderer)

        natural_widths: dict[tuple[int, int], float] = {}
        for r, row in enumerate(self.children):
            if isinstance(row, TableRow):
                for c, cell in enumerate(row.children):
                    if isinstance(cell, TableCell):
                        natural_widths[(r, c)] = cell._dimensions.width

        for r, row in enumerate(self.children):
            if isinstance(row, TableRow):
                for c, cell in enumerate(row.children):
                    if isinstance(cell, TableCell):
                        key = (r, c)
                        if key in original_min_max:
                            cell.min_dimensions = original_min_max[key][0]
                            cell.max_dimensions = original_min_max[key][1]

        table_content_width = self.style.content_box(self._dimensions)[0]
        calculated_widths = self._calculate_column_widths(table_content_width, natural_widths)

        for row in self.children:
            if isinstance(row, TableRow):
                row._calculated_column_widths = calculated_widths

        return super().measure_and_layout(renderer)

    def _calculate_column_widths(
        self, available_width: float, natural_widths: dict[tuple[int, int], float]
    ) -> list[float]:
        """final width for each column."""
        num_cols = self._num_columns
        col_styles = self.column_styles
        widths = [0.0] * num_cols
        is_auto = [False] * num_cols
        is_percent = [False] * num_cols
        percent_values = [0.0] * num_cols
        fixed_width = 0.0
        auto_natural_max = [0.0] * num_cols

        # pass 1: classify columns and compute natural widths for auto cols
        for c in range(num_cols):
            style_width = col_styles[c].width
            if isinstance(style_width, (int, float)):
                widths[c] = float(style_width)
                fixed_width += widths[c]
            elif isinstance(style_width, str) and style_width.endswith("%"):
                try:
                    percent_values[c] = float(style_width[:-1]) / 100.0
                    is_percent[c] = True
                except ValueError:
                    is_auto[c] = True
            else:
                is_auto[c] = True
                max_natural = 0.0
                for r in range(len(self.data)):
                    cell_col_idx = 0
                    row_children = (
                        self.children[r].children
                        if r < len(self.children) and isinstance(self.children[r], TableRow)
                        else []
                    )
                    for cell_idx_in_row, cell_comp in enumerate(row_children):
                        colspan = cell_comp.colspan

                        if cell_col_idx <= c < cell_col_idx + colspan:
                            # spread natural width across auto cols spanned
                            num_auto_spanned = sum(
                                1 for i in range(colspan) if is_auto[cell_col_idx + i]
                            )
                            if num_auto_spanned > 0:
                                natural_w = natural_widths.get((r, cell_idx_in_row), 0)
                                width_per_auto_col = natural_w / num_auto_spanned
                                max_natural = max(max_natural, width_per_auto_col)
                        cell_col_idx += colspan
                auto_natural_max[c] = max_natural
                widths[c] = auto_natural_max[c]

        # pass 2: percentage widths over remaining space
        remaining_width = (
            available_width - fixed_width - sum(widths[c] for c in range(num_cols) if is_auto[c])
        )
        total_percent = sum(percent_values[c] for c in range(num_cols) if is_percent[c])

        if total_percent > 0 and remaining_width > 0:
            unit_width = remaining_width / total_percent
            for c in range(num_cols):
                if is_percent[c]:
                    widths[c] = percent_values[c] * unit_width
        elif total_percent > 0:
            current_total = fixed_width + sum(widths[c] for c in range(num_cols) if is_auto[c])
            if available_width > current_total:
                unit_width = (available_width - current_total) / total_percent
                for c in range(num_cols):
                    if is_percent[c]:
                        widths[c] = percent_values[c] * unit_width
            else:
                for c in range(num_cols):
                    if is_percent[c]:
                        widths[c] = 0

        # pass 3: distribute remaining space among auto columns
        current_total_width = sum(widths)
        extra_space = available_width - current_total_width
        num_auto_cols = sum(1 for c in range(num_cols) if is_auto[c])

        if extra_space > 0 and num_auto_cols > 0:
            total_natural_auto = sum(auto_natural_max[c] for c in range(num_cols) if is_auto[c])
            if total_natural_auto > 0:
                for c in range(num_cols):
                    if is_auto[c]:
                        proportion = auto_natural_max[c] / total_natural_auto
                        widths[c] += extra_space * proportion
            else:
                equal_share = extra_space / num_auto_cols
                for c in range(num_cols):
                    if is_auto[c]:
                        widths[c] += equal_share
        if current_total_width > available_width and current_total_width > 0:
            shrink_ratio = available_width / current_total_width
            widths = [w * shrink_ratio for w in widths]

        return widths
