import numpy as np
import pytest
from lxml import etree

from jeanplot.core.models import Offset, Size
from jeanplot import Container
from jeanplot.core.renderer.svg import SVGRenderer
from jeanplot.core.table import CellStyle, ColumnStyle, Table, TableCell


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
