"""Theme rule cascade applies plot defaults to panel instances."""

import numpy as np

import jeanplot
from jeanplot import (
    PlotData,
    SmoothPanel2D,
    load_plot_theme,
)
from jeanplot.core.style import jstyle


def _data_2d():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(120, 2)).astype(np.float32)
    y = (x[:, :1] + x[:, 1:]).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")


def test_plots_theme_applies_to_smooth_panel_2d():
    load_plot_theme()
    try:
        panel = SmoothPanel2D(plot_data=_data_2d())
        jstyle.apply(panel)
        assert panel.vlim_quantiles == (0.02, 0.98) or list(panel.vlim_quantiles) == [0.02, 0.98]
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)


def test_plots_theme_does_not_clobber_explicit_rescaler():
    """The theme must not overwrite an explicit panel.rescaler.

    Regression: a `PlotPanel: rescaler: ${IdentityRescaler()}` cascade rule
    used to unconditionally override the per-instance rescaler, silently
    breaking biocomp domain figures that pass a real rescaler in YAML.
    The fallback now lives per-`draw()` (no rescaler -> IdentityRescaler),
    so the cascade leaves the field alone.
    """
    from jeanplot import IdentityRescaler

    load_plot_theme()
    try:
        explicit = IdentityRescaler()
        panel = SmoothPanel2D(plot_data=_data_2d(), rescaler=explicit)
        jstyle.apply(panel)
        assert panel.rescaler is explicit
    finally:
        jstyle.clear()
        jeanplot.load_default_theme(force=True)
