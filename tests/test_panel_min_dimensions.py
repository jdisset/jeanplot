"""Tests for content-aware sizing on PlotPanel."""

import numpy as np

import jeanplot
from jeanplot import Figure, PlotData, Size, SmoothPanel2D, SmoothPanel1D, load_plot_theme
from jeanplot.core.style import jstyle


def _data_2d():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(80, 2)).astype(np.float32)
    y = (x[:, :1] + x[:, 1:]).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")


def test_min_dimensions_computed_from_content():
    panel = SmoothPanel2D(plot_data=_data_2d())
    md = panel.min_dimensions
    assert md.width >= panel.axes_size.width
    assert md.height >= panel.axes_size.height


def test_panel_with_colorbar_has_larger_width():
    no_cb = SmoothPanel2D(plot_data=_data_2d(), colorbar_pad=0.0)
    with_cb = SmoothPanel2D(plot_data=_data_2d(), colorbar_pad=0.6)
    no_cb._refresh_content_size()
    with_cb._refresh_content_size()
    assert with_cb.min_dimensions.width > no_cb.min_dimensions.width


def test_explicit_min_dimensions_wins():
    forced = Size(width=5.0, height=5.0)
    panel = SmoothPanel2D(plot_data=_data_2d(), min_dimensions=forced)
    assert panel.min_dimensions.width == 5.0
    assert panel.min_dimensions.height == 5.0
    panel._refresh_content_size()
    assert panel.min_dimensions.width == 5.0


def test_theme_colorbar_pad_reflected_in_min_dims():
    load_plot_theme()
    try:
        panel = SmoothPanel2D(plot_data=_data_2d())
        jstyle.apply(panel)
        assert panel.colorbar_pad >= 0.5
        assert panel.min_dimensions.width >= panel.axes_size.width + panel.colorbar_pad
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)


def test_user_colorbar_pad_wins_over_theme():
    load_plot_theme()
    try:
        panel = SmoothPanel2D(plot_data=_data_2d(), colorbar_pad=2.0)
        jstyle.apply(panel)
        assert panel.colorbar_pad == 2.0
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)


def test_figure_auto_sizes_from_children():
    p1 = SmoothPanel2D(plot_data=_data_2d())
    p2 = SmoothPanel2D(plot_data=_data_2d())
    p3 = SmoothPanel2D(plot_data=_data_2d())
    fig = Figure(p1, p2, p3, layout="row gap=1.0", output_file=None)
    fig.measure_and_layout(None)
    expected_width_min = 3 * p1.min_dimensions.width + 2 * fig.layout.gap
    assert fig._dimensions.width >= expected_width_min - 1e-6
    assert fig._dimensions.height >= p1.min_dimensions.height - 1e-6


def test_smooth_panel_1d_has_legend_pad_via_theme():
    load_plot_theme()
    try:
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, size=(50, 1)).astype(np.float32)
        y = (x * 2).astype(np.float32)
        pd = PlotData(xval=x, yval=y, input_names=["a"], output_name="o")
        panel = SmoothPanel1D(plot_data=pd)
        jstyle.apply(panel)
        assert panel.legend_pad > 0.0
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)
