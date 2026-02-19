from typing import Literal, Any, TypeVar
from pydantic import Field, model_validator, PrivateAttr, BaseModel
import numpy as np

from jeanplot.core.component import Component
from jeanplot.core.container import Container
from jeanplot.core.models import (
    Size,
    BoxStyle,
    LayoutConstraints,
)
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


class TableCell(Container):
    """Represents a single cell within a TableRow."""

    style: CellStyle = Field(default_factory=CellStyle)
    colspan: int = 1
    # rowspan: int = 1 # deferred for simplicity

    # internal attributes set by Table/TableRow
    _row_index: int = PrivateAttr(default=0)
    _col_index: int = PrivateAttr(default=0)
    _is_header: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def apply_styles(self):
        # override parent apply_styles to ensure TableCell's style is CellStyle
        jstyle.apply(self)
        # ensure style is CellStyle after jstyle application
        if not isinstance(self.style, CellStyle):
            # if jstyle replaced it with a BoxStyle, convert it back
            # preserving original cellstyle fields if they existed
            current_style_dict = self.style.model_dump()
            self.style = CellStyle(**current_style_dict)
        return self

    def measure_and_layout(self, renderer=None) -> Size:
        target_width = -1.0  # Flag for no specific width target initially
        if self.style.align_items:
            self.layout.align_items = self.style.align_items
        if self.style.justify_content:
            self.layout.justify_content = self.style.justify_content

        # apply width constraints from parent row before measuring
        if self.parent and hasattr(self.parent, "_calculated_column_widths"):
            col_widths = self.parent._calculated_column_widths
            # Check if _col_index is valid
            if self._col_index < len(col_widths):
                # Check if colspan is valid
                effective_colspan = min(self.colspan, len(col_widths) - self._col_index)
                if effective_colspan > 0:
                    target_width = sum(
                        col_widths[c]
                        for c in range(self._col_index, self._col_index + effective_colspan)
                    )
                    # also account for gaps between spanned columns if border is separate
                    if (
                        effective_colspan > 1
                        and hasattr(self.parent.parent, "border_collapse")
                        and self.parent.parent.border_collapse == "separate"
                        and hasattr(self.parent.parent, "border_spacing")
                    ):
                        target_width += self.parent.parent.border_spacing * (effective_colspan - 1)

                    # apply as min/max dimensions to constrain measurement
                    self.min_dimensions.width = target_width
                    self.max_dimensions.width = target_width

        # apply cell style dimension constraints AFTER column width target
        if self.style.min_width is not None:
            self.min_dimensions.width = max(self.min_dimensions.width, self.style.min_width)
        if self.style.max_width is not None:
            self.max_dimensions.width = min(self.max_dimensions.width, self.style.max_width)
        # If style constraints conflict with target_width, let style win but might break table alignment
        if target_width >= 0 and self.min_dimensions.width > self.max_dimensions.width:
            self._log_debug(
                f"Warning: Cell style min/max width conflicts for cell {self._row_index},{self._col_index}. Using style constraints."
            )
            target_width = -1.0  # Remove target width constraint if style overrides

        if self.style.min_height is not None:
            self.min_dimensions.height = max(self.min_dimensions.height, self.style.min_height)
        if self.style.max_height is not None:
            self.max_dimensions.height = min(self.max_dimensions.height, self.style.max_height)

        # proceed with standard measurement and layout
        # The super() call will measure children naturally, then attempt to layout within constraints
        super().measure_and_layout(renderer)

        # --- Enforce final width ---
        # Ensure final dimensions strictly respect the calculated width constraint if it was set
        # And if style constraints didn't conflict
        if target_width >= 0 and self.min_dimensions.width == self.max_dimensions.width:
            # Check if size differs significantly before forcing
            if abs(self._dimensions.width - target_width) > 1e-6:
                self._dimensions.width = target_width

        # Also enforce max_width if it's less than current width
        elif (
            self.max_dimensions.width < float("inf")
            and self._dimensions.width > self.max_dimensions.width
        ):
            self._dimensions.width = self.max_dimensions.width
        # --- End of width enforcement ---

        return self._dimensions  # Return the potentially adjusted dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """Render the cell background/border, then content."""
        # custom border rendering based on cell style flags
        effective_style = self.style
        draw_top = effective_style.border_top is not False
        draw_right = effective_style.border_right is not False
        draw_bottom = effective_style.border_bottom is not False
        draw_left = effective_style.border_left is not False

        if effective_style.background_color or (
            effective_style.border_color and effective_style.border_width > 0
        ):
            # create a temporary style for rendering the background/border box
            render_style = effective_style.model_copy()

            if not (draw_top and draw_right and draw_bottom and draw_left):
                # if any side is hidden, we can't use the simple rectangle border
                # we'll draw the background only, and draw borders manually
                bg_style = render_style.model_copy()
                bg_style.border_width = 0
                bg_style.border_color = None
                if bg_style.background_color:
                    renderer.render_rectangle(
                        context, self._dimensions, bg_style, matrix, component=self
                    )

                # manually draw border segments if needed
                if render_style.border_color and render_style.border_width > 0:
                    pass  # TODO: implement manual border segment drawing if needed
                    # for now, we just skip the border if any side is hidden
            else:
                # all borders visible, render normally
                renderer.render_rectangle(
                    context, self._dimensions, render_style, matrix, component=self
                )

        if self.debug:
            renderer.render_debug(context, self, matrix)

        # render children (cell content)
        for child in self.children:
            child_matrix = matrix @ child.compute_local_matrix()
            child.render(renderer, context, child_matrix)


class TableRow(Container):
    """Represents a single row within a Table."""

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row",
            align_items="stretch",  # stretch cells vertically by default
            justify_content="start",
            gap=0,  # gap usually handled by cell padding/margin or table border spacing
        )
    )
    # internal attributes set by Table
    _row_index: int = PrivateAttr(default=0)
    _is_header: bool = PrivateAttr(default=False)
    _calculated_column_widths: list[float] = PrivateAttr(default_factory=list)


class Table(Container):
    """Container that arranges data into rows and columns."""

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="column",
            align_items="stretch",  # stretch rows horizontally
            justify_content="start",
            gap=0,  # vertical gap handled by cell padding/margin or border spacing
        )
    )
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(
            # default table style often includes an outer border
            border_color="#333333",
            border_width=0.5,
        )
    )

    data: list[list[Any]] = Field(default_factory=list)
    column_styles: list[ColumnStyle] = Field(default_factory=list)
    header_rows: int = 0

    border_collapse: Literal["collapse", "separate"] = "collapse"
    border_spacing: float = 0

    _num_columns: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def build_table(self):
        self.children = []  # clear any existing children
        if not self.data:
            self._num_columns = 0  # Explicitly set for empty data case
            return self

        max_effective_cols = 0
        for row_data in self.data:
            current_effective_cols = 0
            for cell_content in row_data:
                colspan = 1
                # check if it's a TableCell instance and get its colspan
                if isinstance(cell_content, TableCell):
                    colspan = cell_content.colspan
                # or check if it's a dict that might represent a TableCell (less robust)
                elif isinstance(cell_content, dict) and "colspan" in cell_content:
                    colspan = cell_content.get("colspan", 1)

                current_effective_cols += colspan
            max_effective_cols = max(max_effective_cols, current_effective_cols)
        self._num_columns = max_effective_cols

        # ensure column_styles has enough entries
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

            col_offset = 0  # track column index considering colspan
            for c_nominal, cell_content in enumerate(row_data):
                c_actual = col_offset
                # ensure we don't process beyond the calculated number of columns
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
                    # default: wrap content in a Text component
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

                merged_style_dict = self._merge_styles(
                    base_style.model_dump(), cell.style.model_dump()
                )
                cell.style = CellStyle(**merged_style_dict)

                table_row.add_child(cell)

                col_offset += cell.colspan

            # handle row padding/border spacing for 'separate' mode
            if self.border_collapse == "separate" and self.border_spacing > 0:
                table_row.layout.gap = self.border_spacing
                # add padding to the row itself to create vertical spacing
                table_row.style.padding = (self.border_spacing / 2, 0, self.border_spacing / 2, 0)

            self.add_child(table_row)  # add completed row to table

        # apply table-level border spacing padding if separate
        if self.border_collapse == "separate" and self.border_spacing > 0:
            pad_t, pad_r, pad_b, pad_l = self.style.padding
            self.style.padding = (
                pad_t + self.border_spacing / 2,
                pad_r + self.border_spacing / 2,
                pad_b + self.border_spacing / 2,
                pad_l + self.border_spacing / 2,
            )
            self.layout.gap = self.border_spacing

        return self

    def _merge_styles(self, base: dict, overlay: dict) -> dict:
        """Merges two style dictionaries (overlay takes precedence). Handles None."""
        merged = base.copy()
        for key, value in overlay.items():
            if value is not None:  # only override if overlay value is explicitly set
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_styles(merged[key], value)
                else:
                    merged[key] = value
        return merged

    def measure_and_layout(self, renderer=None) -> Size:
        """Overrides container layout to handle column widths."""
        if not self.children:  # no rows
            return super().measure_and_layout(renderer)

        # measure natural cell sizes
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
                        # allow natural measurement
                        cell.min_dimensions.width = 0
                        cell.max_dimensions.width = float("inf")

        super().measure_and_layout(renderer)

        natural_widths: dict[tuple[int, int], float] = {}
        for r, row in enumerate(self.children):
            if isinstance(row, TableRow):
                for c, cell in enumerate(row.children):
                    if isinstance(cell, TableCell):
                        natural_widths[(r, c)] = cell._dimensions.width

        # restore original min/max constraints
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

        # apply calculated widths and re-layout
        for row in self.children:
            if isinstance(row, TableRow):
                row._calculated_column_widths = calculated_widths  # pass widths to row/cells

        final_size = super().measure_and_layout(renderer)

        return final_size

    def _calculate_column_widths(
        self, available_width: float, natural_widths: dict[tuple[int, int], float]
    ) -> list[float]:
        """Determine the final width for each column."""
        num_cols = self._num_columns
        col_styles = self.column_styles
        widths = [0.0] * num_cols
        is_auto = [False] * num_cols
        is_percent = [False] * num_cols
        percent_values = [0.0] * num_cols
        fixed_width = 0.0
        auto_natural_max = [0.0] * num_cols

        # --- Pass 1: Identify types and fixed/natural sizes ---
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
                    is_auto[c] = True  # treat invalid percentage as auto
            else:  # 'auto' or invalid
                is_auto[c] = True
                # find max natural width for this auto column (considering colspan)
                max_natural = 0.0
                for r in range(len(self.data)):
                    cell_col_idx = 0
                    for cell_idx_in_row, cell_data in enumerate(self.data[r]):
                        cell_comp = self.children[r].children[
                            cell_idx_in_row
                        ]  # get the actual cell component
                        colspan = cell_comp.colspan

                        # is this cell part of the current column 'c'?
                        if cell_col_idx <= c < cell_col_idx + colspan:
                            # if cell spans multiple auto columns, distribute natural width
                            num_auto_spanned = sum(
                                1 for i in range(colspan) if is_auto[cell_col_idx + i]
                            )
                            if num_auto_spanned > 0:
                                natural_w = natural_widths.get((r, cell_idx_in_row), 0)
                                width_per_auto_col = natural_w / num_auto_spanned
                                max_natural = max(max_natural, width_per_auto_col)
                        cell_col_idx += colspan
                auto_natural_max[c] = max_natural
                widths[c] = auto_natural_max[c]  # start with natural max

        # --- Pass 2: Calculate percentage widths ---
        remaining_width = (
            available_width - fixed_width - sum(widths[c] for c in range(num_cols) if is_auto[c])
        )
        total_percent = sum(percent_values[c] for c in range(num_cols) if is_percent[c])

        if total_percent > 0 and remaining_width > 0:
            # calculate based on proportion of remaining space
            unit_width = remaining_width / total_percent
            for c in range(num_cols):
                if is_percent[c]:
                    widths[c] = percent_values[c] * unit_width
        elif total_percent > 0:  # not enough space, distribute available proportionally
            # re-calculate total fixed + auto
            current_total = fixed_width + sum(widths[c] for c in range(num_cols) if is_auto[c])
            if available_width > current_total:
                unit_width = (available_width - current_total) / total_percent
                for c in range(num_cols):
                    if is_percent[c]:
                        widths[c] = percent_values[c] * unit_width
            else:  # no space left for percentages
                for c in range(num_cols):
                    if is_percent[c]:
                        widths[c] = 0  # or some minimum?

        # --- Pass 3: Distribute remaining space for 'auto' columns (if any) ---
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
                # distribute equally if natural sizes are zero
                equal_share = extra_space / num_auto_cols
                for c in range(num_cols):
                    if is_auto[c]:
                        widths[c] += equal_share
        # TODO: handle shrinkage if current_total_width > available_width

        return widths
