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


def test_panel_with_padding_has_larger_width():
    no_pad = SmoothPanel2D(plot_data=_data_2d())
    with_pad = SmoothPanel2D(plot_data=_data_2d(), style={"padding": {"right": 0.6}})
    assert with_pad.min_dimensions.width > no_pad.min_dimensions.width


def test_explicit_min_dimensions_wins():
    forced = Size(width=5.0, height=5.0)
    panel = SmoothPanel2D(plot_data=_data_2d(), min_dimensions=forced)
    assert panel.min_dimensions.width == 5.0
    assert panel.min_dimensions.height == 5.0


def test_theme_padding_right_reflected_in_min_dims():
    load_plot_theme()
    try:
        panel = SmoothPanel2D(plot_data=_data_2d())
        jstyle.apply(panel)
        assert panel.style.padding.right >= 0.5
        assert panel.min_dimensions.width >= panel.axes_size.width + panel.style.padding.right
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)


def test_user_padding_right_wins_over_theme():
    load_plot_theme()
    try:
        panel = SmoothPanel2D(plot_data=_data_2d(), style={"padding": {"right": 2.0}})
        jstyle.apply(panel)
        assert panel.style.padding.right == 2.0
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


def test_padding_sums_to_min_dimensions():
    # draw_colorbar=False so effective_padding == style.padding (no out-of-axes colorbar
    # reserve folded into the right inset); this test is about the padding arithmetic.
    panel = SmoothPanel2D(
        plot_data=_data_2d(),
        axes_size=Size(width=2.5, height=2.0),
        title=None,
        draw_colorbar=False,
        style={"padding": {"left": 0.5, "right": 0.6, "bottom": 0.5, "top": 0.0}},
    )
    p = panel.effective_padding
    assert (p.left, p.top, p.right, p.bottom) == (0.5, 0.0, 0.6, 0.5)
    assert panel.min_dimensions.width == panel.axes_size.width + p.left + p.right
    assert panel.min_dimensions.height == panel.axes_size.height + p.top + p.bottom


def test_colorbar_reserved_in_effective_padding():
    # the out-of-axes colorbar (axes-fraction position+size > 1) is auto-reserved as
    # right inset so the layout accounts for it; toggling draw_colorbar removes it.
    common = dict(plot_data=_data_2d(), axes_size=Size(width=2.5, height=2.0), title=None)
    with_cb = SmoothPanel2D(draw_colorbar=True, **common)
    without = SmoothPanel2D(draw_colorbar=False, **common)
    assert with_cb.effective_padding.right > without.effective_padding.right
    assert with_cb.min_dimensions.width == with_cb.axes_size.width + with_cb.effective_padding.right


def test_effective_padding_reserves_title_room():
    untitled = SmoothPanel2D(plot_data=_data_2d(), title=None)
    titled = SmoothPanel2D(plot_data=_data_2d(), title="hello")
    assert titled.effective_padding.top == untitled.effective_padding.top + 0.3
    assert titled.min_dimensions.height == untitled.min_dimensions.height + 0.3


def test_smooth_panel_1d_has_padding_right_via_theme():
    load_plot_theme()
    try:
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, size=(50, 1)).astype(np.float32)
        y = (x * 2).astype(np.float32)
        pd = PlotData(xval=x, yval=y, input_names=["a"], output_name="o")
        panel = SmoothPanel1D(plot_data=pd)
        jstyle.apply(panel)
        assert panel.style.padding.right > 0.0
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)
