"""Smoke tests: each panel constructs, draws on an Axes, render_txt() works."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jeanplot import (
    AsciiHeatmapPanel,
    DensityPanel1D,
    Figure,
    GradientFieldPanel2D,
    GridHistogramPanel,
    IdentityLineOverlay,
    MVPPanel,
    ParticlePanel,
    PlotData,
    SmoothGradMagnitudePanel2D,
    SmoothPanel1D,
    SmoothPanel2D,
    SmoothPanel3D,
    StackedPolyPanel,
    ViolinPanel,
)
from jeanplot.panels.auto import auto_panel


@pytest.fixture
def data_1d():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1)).astype(np.float32)
    y = (np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=x.shape)).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a"], output_name="out")


@pytest.fixture
def data_2d():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(400, 2)).astype(np.float32)
    y = (x[:, :1] * x[:, 1:] + 0.1 * rng.normal(size=(400, 1))).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="out")


@pytest.fixture
def data_3d():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(500, 3)).astype(np.float32)
    y = (x.sum(axis=1, keepdims=True) / 3.0 + 0.1 * rng.normal(size=(500, 1))).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a", "b", "c"], output_name="out")


@pytest.fixture
def axes():
    fig, ax = plt.subplots(figsize=(4, 3))
    yield ax
    plt.close(fig)


def test_smooth_panel_1d_constructs_and_draws(data_1d, axes):
    panel = SmoothPanel1D(plot_data=data_1d, knn_stats_params={"k": 20, "radius": 0.2})
    panel.draw(axes)


def test_smooth_panel_2d_constructs_and_draws(data_2d, axes):
    panel = SmoothPanel2D(
        plot_data=data_2d,
        knn_grid_params={"grid_resolution": 32, "knn_stats_params": {"k": 30, "radius": 0.15}},
    )
    panel.draw(axes)


def test_smooth_grad_magnitude_2d(data_2d, axes):
    panel = SmoothGradMagnitudePanel2D(
        plot_data=data_2d,
        knn_grid_params={"grid_resolution": 24, "knn_stats_params": {"k": 30, "radius": 0.15}},
    )
    panel.draw(axes)


def test_gradient_field_2d(data_2d, axes):
    panel = GradientFieldPanel2D(
        plot_data=data_2d,
        knn_grid_params={"grid_resolution": 24, "knn_stats_params": {"k": 30, "radius": 0.15}},
    )
    panel.draw(axes)


def test_smooth_panel_3d_is_layout_only_container(data_3d):
    panel = SmoothPanel3D(plot_data=data_3d)
    assert panel.is_drawable is False
    assert len(panel.children) == 2
    assert panel.draw(None) is None


def test_mvp_panel_constructs_and_draws(axes):
    rng = np.random.default_rng(0)
    measured = rng.uniform(0, 1, size=300)
    predicted = measured + 0.1 * rng.normal(size=300)
    panel = MVPPanel(
        measured=measured,
        predicted=predicted,
        show_density=False,
        show_trendline=False,
        show_stats=False,
        show_noise_floor=False,
    )
    panel.draw(axes)


def test_density_panel_1d_constructs_and_draws(axes):
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    panel = DensityPanel1D(values=values, color="blue")
    panel.draw(axes)


def test_grid_histogram_panel_constructs_and_draws(data_1d, axes):
    panel = GridHistogramPanel(plot_data=data_1d, res=50, draw_colorbar=False)
    panel.draw(axes)


def test_violin_panel_constructs(data_2d):
    panel = ViolinPanel(plot_data=data_2d, grid_resolution=16)
    assert panel.plot_data.x.shape == (400, 2)


def test_particle_panel_constructs_and_draws(axes):
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1, size=(3, 30))
    panel = ParticlePanel(data=data, value_names=["a", "b", "c"])
    panel.draw(axes)


def test_stacked_poly_panel_constructs_and_draws(axes):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=x.shape)
    panel = StackedPolyPanel(x=x, y=y, quantiles=(0.25, 0.5, 0.75))
    panel.draw(axes)


def test_ascii_heatmap_renders_txt():
    data = np.random.default_rng(0).uniform(0, 1, size=(20, 40))
    panel = AsciiHeatmapPanel(data=data, xres=40, yres=10, show_colorbar=False)
    txt = panel.render_txt()
    assert txt is not None
    assert isinstance(txt, str)
    assert len(txt) > 0


def test_identity_line_overlay_draws(axes):
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    overlay = IdentityLineOverlay()
    overlay.draw(axes)


def test_auto_panel_dispatch_1d(data_1d):
    panel = auto_panel(data_1d)
    assert isinstance(panel, SmoothPanel1D)


def test_auto_panel_dispatch_2d(data_2d):
    panel = auto_panel(data_2d)
    assert isinstance(panel, SmoothPanel2D)


def test_auto_panel_dispatch_3d(data_3d):
    panel = auto_panel(data_3d)
    assert isinstance(panel, SmoothPanel3D)


def test_auto_panel_force_dim(data_2d):
    panel = auto_panel(data_2d, force_dim=1)
    assert isinstance(panel, SmoothPanel1D)


def test_auto_panel_rejects_unsupported_dim(data_1d):
    with pytest.raises(ValueError):
        auto_panel(data_1d, force_dim=5)


def test_panel_by_dim_dispatch_table_complete():
    from jeanplot.panels.auto import _PANEL_BY_DIM

    assert set(_PANEL_BY_DIM.keys()) == {1, 2, 3}
    assert _PANEL_BY_DIM[1] is SmoothPanel1D
    assert _PANEL_BY_DIM[2] is SmoothPanel2D
    assert _PANEL_BY_DIM[3] is SmoothPanel3D


def test_render_txt_1d_default_none(axes, data_1d):
    panel = SmoothPanel1D(plot_data=data_1d)
    txt = panel.render_txt()
    assert txt is None or isinstance(txt, str)


def test_render_txt_2d_default_none(data_2d):
    panel = SmoothPanel2D(plot_data=data_2d)
    txt = panel.render_txt()
    assert txt is None or isinstance(txt, str)


def test_figure_with_smooth_panel_constructs(data_2d, tmp_path):
    fig = Figure(
        output_dir=str(tmp_path),
        output_file="smooth_2d.png",
        children=[
            SmoothPanel2D(
                plot_data=data_2d,
                knn_grid_params={
                    "grid_resolution": 16,
                    "knn_stats_params": {"k": 20, "radius": 0.2},
                },
                draw_colorbar=False,
            )
        ],
    )
    assert fig.output_path.name == "smooth_2d.png"
