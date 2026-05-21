"""compose.py tree-construction helpers."""

import numpy as np

from jeanplot import (
    Container,
    PlotData,
    SmoothPanel1D,
    SmoothPanel2D,
    panel_grid,
    panel_row,
    panels_from_datas,
    build_figure_metadata,
)


def _pd(dim: int):
    rng = np.random.default_rng(dim)
    x = rng.uniform(0, 1, size=(50, dim)).astype(np.float32)
    y = rng.uniform(0, 1, size=(50, 1)).astype(np.float32)
    return PlotData(
        xval=x,
        yval=y,
        input_names=[f"in{i}" for i in range(dim)],
        output_name="out",
        metadata={"network_name": f"n{dim}"},
    )


def test_panel_row_wraps_panels_in_row_container():
    p1 = SmoothPanel1D(plot_data=_pd(1))
    p2 = SmoothPanel2D(plot_data=_pd(2))
    row = panel_row([p1, p2], gap=4.0)
    assert isinstance(row, Container)
    assert row.layout.direction == "row"
    assert row.layout.gap == 4.0
    assert list(row.children) == [p1, p2]


def test_panel_row_weights():
    p1 = SmoothPanel1D(plot_data=_pd(1))
    p2 = SmoothPanel2D(plot_data=_pd(2))
    row = panel_row([p1, p2], weights=[1.0, 2.0])
    assert row.layout.main_axis_weights == [1.0, 2.0]


def test_panel_grid_column_then_rows():
    p11 = SmoothPanel1D(plot_data=_pd(1))
    p12 = SmoothPanel2D(plot_data=_pd(2))
    p21 = SmoothPanel1D(plot_data=_pd(1))
    grid = panel_grid([[p11, p12], [p21]])
    assert grid.layout.direction == "column"
    assert len(grid.children) == 2
    assert grid.children[0].layout.direction == "row"
    assert len(grid.children[0].children) == 2
    assert len(grid.children[1].children) == 1


def test_panels_from_datas_dispatches_by_dim():
    panels = panels_from_datas([_pd(1), _pd(2)])
    assert isinstance(panels[0], SmoothPanel1D)
    assert isinstance(panels[1], SmoothPanel2D)


def test_build_figure_metadata_aggregates():
    p1 = SmoothPanel1D(plot_data=_pd(1))
    p2 = SmoothPanel2D(plot_data=_pd(2))
    md = build_figure_metadata([p1, p2], extra={"k": "v"})
    assert md["network_name"] == "n2"  # later panel wins
    assert md["k"] == "v"
