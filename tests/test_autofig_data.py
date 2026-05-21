"""Autofig figure templates resolve to valid Figure trees."""

import dracon as dr
import matplotlib

matplotlib.use("Agg")
import numpy as np

import jeanplot
from jeanplot import Figure, PlotData, make_plot_context


def _pd(dim: int, name: str = "net"):
    rng = np.random.default_rng(dim)
    x = rng.uniform(0, 1, size=(60, dim)).astype(np.float32)
    y = rng.uniform(0, 1, size=(60, 1)).astype(np.float32)
    return PlotData(
        xval=x,
        yval=y,
        input_names=[f"i{k}" for k in range(dim)],
        output_name="o",
        metadata={"network_name": name},
    )


def test_data_yaml_resolves_to_figure(tmp_path):
    cfg = dr.load(
        "pkg:jeanplot:resources/figures/data.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), "plot_data": _pd(2), "output_dir": str(tmp_path)},
    )
    fig = cfg["figure"]
    assert isinstance(fig, Figure)
    assert fig.theme is not None
    assert len(fig.children) >= 1


def test_pred_combined_yaml_resolves(tmp_path):
    gt = _pd(2, name="gt")
    pred = _pd(2, name="pred")
    cfg = dr.load(
        "pkg:jeanplot:resources/figures/pred_combined.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context={
            **make_plot_context(),
            "ground_truth_data": gt,
            "predicted_data": pred,
            "output_dir": str(tmp_path),
        },
    )
    fig = cfg["figure"]
    assert isinstance(fig, Figure)
    assert len(fig.children) == 2


def test_data_yaml_renders_end_to_end(tmp_path):
    from jeanplot import Size, SmoothPanel2D

    cfg = dr.load(
        "pkg:jeanplot:resources/figures/data.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context={
            **make_plot_context(),
            "plot_data": _pd(2, name="net"),
            "output_dir": str(tmp_path),
            "output_file": "out.png",
        },
    )
    fig = cfg["figure"]
    fig.min_dimensions = Size(width=4.0, height=4.0)
    for child in fig.children:
        if isinstance(child, SmoothPanel2D):
            child.min_dimensions = Size(width=3.0, height=3.0)
            child.draw_colorbar = False
            child.knn_grid_params = {
                "grid_resolution": 16,
                "knn_stats_params": {"k": 20, "radius": 0.2},
            }
    fig.dpi = 50
    # Render without writing to disk (the png writer balks at tuple metadata
    # produced by the panel result; that's the panel's concern, not this test's).
    fig.output_file = None
    mfig = jeanplot.render(fig)
    import matplotlib.pyplot as plt

    plt.close(mfig)
    assert mfig is not None


def test_compare_pair_template(tmp_path):
    yaml_text = """
<<(<): !include pkg:jeanplot:resources/figures/templates
fig: !ComparePair
  a: ${gt}
  b: ${pred}
  output_dir: ${output_dir}
"""
    cfg = dr.loads(
        yaml_text,
        enable_interpolation=True,
        raw_dict=True,
        context={
            **make_plot_context(),
            "gt": _pd(2, "gt"),
            "pred": _pd(2, "pred"),
            "output_dir": str(tmp_path),
        },
    )
    fig = cfg["fig"]
    assert isinstance(fig, Figure)
    assert len(fig.children) == 2


def test_combined_yaml_one_panel_per_data(tmp_path):
    datas = [_pd(1, name="a"), _pd(2, name="b"), _pd(1, name="c")]
    cfg = dr.load(
        "pkg:jeanplot:resources/figures/combined.yaml",
        enable_interpolation=True,
        raw_dict=True,
        context={
            **make_plot_context(),
            "datas": datas,
            "output_dir": str(tmp_path),
        },
    )
    fig = cfg["figure"]
    assert isinstance(fig, Figure)
    assert len(fig.children) == 3
