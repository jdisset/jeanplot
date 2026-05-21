"""plots.yaml SSOT theme loads cleanly and exposes a rule cascade."""

import dracon as dr

from jeanplot import make_plot_context


def test_plots_theme_loads():
    cfg = dr.load(
        "pkg:jeanplot:resources/themes/plots.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context=make_plot_context(),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    assert "rules" in cfg
    assert cfg["rules"] is not None


def test_plots_theme_vars_overridable():
    cfg = dr.load(
        "pkg:jeanplot:resources/themes/plots.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), "xlims": [0.1, 0.9]},
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    assert cfg["rules"] is not None


def test_paper_theme_loads():
    cfg = dr.load(
        "pkg:jeanplot:resources/themes/paper.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context=make_plot_context(),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    assert "rules" in cfg
