"""AutoPanel dim-dispatch: Python helper and YAML template agree."""

import dracon as dr
import numpy as np

from jeanplot import (
    PlotData,
    SmoothPanel1D,
    SmoothPanel2D,
    SmoothPanel3D,
    auto_panel,
    make_plot_context,
)


def _pd(dim: int):
    rng = np.random.default_rng(dim)
    x = rng.uniform(0, 1, size=(40, dim)).astype(np.float32)
    y = rng.uniform(0, 1, size=(40, 1)).astype(np.float32)
    return PlotData(
        xval=x,
        yval=y,
        input_names=[f"i{k}" for k in range(dim)],
        output_name="o",
    )


def test_auto_panel_python_dispatch_1d():
    assert isinstance(auto_panel(_pd(1)), SmoothPanel1D)


def test_auto_panel_python_dispatch_2d():
    assert isinstance(auto_panel(_pd(2)), SmoothPanel2D)


def test_auto_panel_python_dispatch_3d():
    assert isinstance(auto_panel(_pd(3)), SmoothPanel3D)


def test_auto_panel_yaml_template_dispatch_2d():
    """Use AutoPanel as a tag from a wrapper YAML — the template is propagated."""
    yaml = """
<<(<): !include pkg:jeanplot:resources/templates/auto_panel
panel: !AutoPanel
  plot_data: ${pd}
"""
    cfg = dr.loads(
        yaml,
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), "pd": _pd(2)},
    )
    assert isinstance(cfg["panel"], SmoothPanel2D)


def test_auto_panel_yaml_template_dispatch_1d():
    yaml = """
<<(<): !include pkg:jeanplot:resources/templates/auto_panel
panel: !AutoPanel
  plot_data: ${pd}
"""
    cfg = dr.loads(
        yaml,
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), "pd": _pd(1)},
    )
    assert isinstance(cfg["panel"], SmoothPanel1D)


def test_auto_panel_yaml_template_dispatch_3d():
    yaml = """
<<(<): !include pkg:jeanplot:resources/templates/auto_panel
panel: !AutoPanel
  plot_data: ${pd}
"""
    cfg = dr.loads(
        yaml,
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), "pd": _pd(3)},
    )
    assert isinstance(cfg["panel"], SmoothPanel3D)
