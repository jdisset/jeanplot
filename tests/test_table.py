import numpy as np
import pytest
from lxml import etree

from jeanplot.core.models import LayoutConstraints, Offset, Size
from jeanplot import Container
from jeanplot.core.renderer.svg import SVGRenderer
from jeanplot.core.table import CellStyle, ColumnStyle, Table, TableCell
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.figure import Figure


class _SizedPanel(PlotPanel):
    def draw(self, ax):  # noqa: D401
        pass


_SizedPanel.model_rebuild(force=True)


def test_table_survives_figure_layout_cycle():
    """A Table nested in a Figure keeps its aligned column widths.

    Regression: the Figure's measure/layout cycle reassigned the Table's `parent`
    each pass, re-firing build_table (which rebuilt the rows and wiped the computed
    column widths), collapsing every cell to width 0. Guarded now by the parent
    identity check + idempotent build_table.
    """
    def cells():
        return [
            _SizedPanel(axes_size=Size(width=1.6, height=1.0)),
            _SizedPanel(axes_size=Size(width=2.4, height=1.0)),
        ]

    table = Table(
        data=[cells(), cells(), cells()],
        column_styles=[ColumnStyle(width="auto"), ColumnStyle(width="auto")],
    )
    fig = Figure(children=[table], layout=LayoutConstraints(direction="column"))
    fig.measure_and_layout(None)

    rows = table.children
    assert all(len(r.children) == 2 for r in rows)
    widths = [[round(c._dimensions.width, 3) for c in r.children] for r in rows]
    # every cell has real width, and column c is identical across all rows
    assert all(w[0] > 0 and w[1] > 0 for w in widths)
    assert len({w[0] for w in widths}) == 1
    assert len({w[1] for w in widths}) == 1


def test_table_build_is_idempotent_under_revalidation():
    """Reassigning an unrelated field doesn't rebuild rows (preserves identity)."""
    table = Table(data=[["a", "b"], ["c", "d"]])
    rows_before = list(table.children)
    table.id = "tbl"  # triggers validate_assignment -> all model_validators re-run
    assert [id(r) for r in table.children] == [id(r) for r in rows_before]


def test_table_column_widths_shrink_when_overflowing():
    table = Table(
        data=[["a", "b"]],
        column_styles=[ColumnStyle(width=100), ColumnStyle(width=100)],
    )

    widths = table._calculate_column_widths(available_width=100, natural_widths={})

    assert widths[0] == pytest.approx(50.0)
    assert widths[1] == pytest.approx(50.0)
    assert sum(widths) == pytest.approx(100.0)


def test_table_constructs_without_recursion_and_builds_rows():
    table = Table(data=[["a", "b"], ["c", "d"]])

    assert len(table.children) == 2
    assert all(len(row.children) == 2 for row in table.children)


def test_table_column_widths_handle_truncated_rows():
    cell = TableCell(colspan=3)
    table = Table(
        data=[[cell, "ignored"]],
        column_styles=[ColumnStyle(width="auto"), ColumnStyle(width="auto"), ColumnStyle(width="auto")],
    )
    widths = table._calculate_column_widths(available_width=100, natural_widths={(0, 0): 90.0})

    assert len(widths) == table._num_columns
    assert sum(widths) <= 100.000001


def test_table_cell_partial_borders_render_selected_edges():
    renderer = SVGRenderer()
    root = renderer.create_context(width=100, height=100)

    cell = TableCell(
        id="cell",
        min_dimensions=Size(width=20, height=10),
        style=CellStyle(
            border_color="#000000",
            border_width=1.0,
            border_top=True,
            border_right=False,
            border_bottom=True,
            border_left=True,
        ),
    )
    cell.measure_and_layout(renderer)
    cell.render(renderer, root, np.eye(3))

    svg = renderer.render_to_output(root)
    xml = etree.fromstring(svg.encode())
    paths = xml.xpath("//*[local-name()='path']")

    assert len(paths) == 1
    d = paths[0].get("d", "")
    assert "M 0 0 L 20.000 0" in d
    assert "M 0 10.000 L 20.000 10.000" in d
    assert "M 0 0 L 0 10.000" in d
    assert "M 20.000 0 L 20.000 10.000" not in d


def test_table_cell_render_uses_world_matrix_for_attached_children():
    renderer = SVGRenderer()
    root = renderer.create_context(width=100, height=100)

    anchor = Container(
        id="anchor",
        min_dimensions=Size(width=10, height=10),
        style=CellStyle(background_color="#ff0000"),
    )
    attached = Container(
        id="attached",
        min_dimensions=Size(width=5, height=5),
        style=CellStyle(background_color="#00ff00"),
        attached_to="anchor",
        attachment_offset=Offset(reference_relative=(1.0, 0.0)),
    )
    cell = TableCell(min_dimensions=Size(width=40, height=20), children=[anchor, attached])
    cell.measure_and_layout(renderer)
    cell.render(renderer, root, np.eye(3))

    svg = renderer.render_to_output(root)
    xml = etree.fromstring(svg.encode())
    attached_group = xml.find(".//*[@id='attached']")
    assert attached_group is not None
    assert "10.000000" in attached_group.get("transform", "")
