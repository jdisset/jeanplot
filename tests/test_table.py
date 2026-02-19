import numpy as np
import pytest
from lxml import etree

from jeanplot.core.models import Size
from jeanplot.core.renderer.svg import SVGRenderer
from jeanplot.core.table import CellStyle, ColumnStyle, Table, TableCell


def test_table_column_widths_shrink_when_overflowing():
    table = Table.model_construct(
        data=[["a", "b"]],
        column_styles=[ColumnStyle(width=100), ColumnStyle(width=100)],
        children=[],
    )
    table._num_columns = 2

    widths = table._calculate_column_widths(available_width=100, natural_widths={})

    assert widths[0] == pytest.approx(50.0)
    assert widths[1] == pytest.approx(50.0)
    assert sum(widths) == pytest.approx(100.0)


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
