"""Paper theme acts as a default supplier under cascade-fill.

The headline invariant of refactor 04 on the jeanplot side: paper.yaml carries
domain defaults (sizing, colormap, vlim floor/range) that get applied to panels
that don't declare them explicitly, but explicit per-panel values still win.
"""

import dracon as dr

from jeanplot import (
    DEFAULT_TYPES,
    SmoothPanel2D,
    SmoothPanel1D,
    SmoothGradMagnitudePanel2D,
    jstyle,
    make_context_from_types,
)


def _load_paper_rules(extra_ctx: dict | None = None):
    ctx = make_context_from_types(DEFAULT_TYPES)
    if extra_ctx:
        ctx.update(extra_ctx)
    cfg = dr.load(
        "pkg:jeanplot:resources/themes/paper.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context=ctx,
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    return cfg["rules"]


def test_paper_theme_fills_smooth2d_defaults():
    jstyle.update(_load_paper_rules())
    panel = SmoothPanel2D()
    jstyle.apply(panel)
    assert panel.vlim_min_floor == 0.0
    assert panel.vlim_min_range == 0.1
    assert panel.colorbar_pad == 0.6
    assert panel.heatmap_params["cmap"] == "bc_blues"


def test_paper_theme_does_not_clobber_explicit_values():
    jstyle.update(_load_paper_rules())
    panel = SmoothPanel2D(
        vlim_min_floor=0.42,
        colorbar_pad=0.1,
        heatmap_params={"cmap": "magma"},
    )
    jstyle.apply(panel)
    assert panel.vlim_min_floor == 0.42
    assert panel.colorbar_pad == 0.1
    assert panel.heatmap_params["cmap"] == "magma"


def test_paper_theme_grad_panel_default_cmap():
    jstyle.update(_load_paper_rules())
    panel = SmoothGradMagnitudePanel2D()
    jstyle.apply(panel)
    assert panel.heatmap_params["cmap"] == "bc_reds"
    assert panel.colorbar_pad == 0.6


def test_paper_theme_smooth1d_legend_pad_default():
    jstyle.update(_load_paper_rules())
    panel = SmoothPanel1D()
    jstyle.apply(panel)
    assert panel.legend_pad == 1.2


def test_paper_theme_smooth1d_legend_pad_user_wins():
    jstyle.update(_load_paper_rules())
    panel = SmoothPanel1D(legend_pad=0.4)
    jstyle.apply(panel)
    assert panel.legend_pad == 0.4


def test_paper_theme_vlim_min_floor_overridable_via_context():
    jstyle.update(_load_paper_rules(extra_ctx={"vlim_min_floor": 0.05, "vlim_min_range": 0.2}))
    panel = SmoothPanel2D()
    jstyle.apply(panel)
    assert panel.vlim_min_floor == 0.05
    assert panel.vlim_min_range == 0.2


def test_paper_theme_axes_size_default_from_paper_axes_size():
    jstyle.update(_load_paper_rules(extra_ctx={"paper_axes_size": [3.0, 2.5]}))
    panel = SmoothPanel2D()
    jstyle.apply(panel)
    assert panel.axes_size.width == 3.0
    assert panel.axes_size.height == 2.5
