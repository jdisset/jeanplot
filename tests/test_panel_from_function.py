"""Tests for `panel_from` decorator."""

from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jeanplot import PlotData, make_plot_context
from jeanplot.data import IdentityRescaler, PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.panels.from_function import panel_from
from jeanplot.panels.smooth_2d import SmoothPanel2D


def _toy_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(80, 2)).astype(np.float32)
    y = (x[:, :1] + x[:, 1:]).astype(np.float32)
    return PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")


def _toy_fn(
    X,
    Y,
    input_names,
    output_name,
    rescaler,
    ax,
    threshold: float = 0.5,
    mode: Literal["fast", "accurate"] = "fast",
    annotate: bool = True,
    extras: dict | None = None,
):
    ax.plot(X[:, 0], Y[:, 0])
    return PlotFunctionResult(rendering=None, metadata={"threshold": threshold, "mode": mode})


ToyPanel = panel_from(_toy_fn, name="ToyPanel")


def test_generated_class_has_expected_fields():
    fields = ToyPanel.model_fields
    assert "threshold" in fields
    assert "mode" in fields
    assert "annotate" in fields
    assert "extras" in fields
    for pd_key in ("X", "Y", "input_names", "output_name"):
        assert pd_key not in fields
    assert "ax" not in fields


def test_defaults_propagate_from_signature():
    p = ToyPanel(plot_data=_toy_data())
    assert p.threshold == 0.5
    assert p.mode == "fast"
    assert p.annotate is True
    assert p.extras is None


def test_inherited_plotpanel_fields_present():
    fields = ToyPanel.model_fields
    for inherited in ("plot_data", "rescaler", "title", "xtitle"):
        assert inherited in fields


def test_panel_is_plotpanel_subclass():
    assert issubclass(ToyPanel, PlotPanel)


def test_draw_calls_function_and_records_mappable():
    pd = _toy_data()
    panel = ToyPanel(plot_data=pd, threshold=0.9)
    _, ax = plt.subplots()
    result = panel.draw(ax)
    plt.close()
    assert isinstance(result, PlotFunctionResult)
    assert result.metadata["threshold"] == 0.9


def test_rescaler_fallback_to_identity():
    pd = _toy_data()
    panel = ToyPanel(plot_data=pd)
    assert panel.rescaler is None
    _, ax = plt.subplots()
    panel.draw(ax)
    plt.close()


def test_drawing_function_is_callable_standalone():
    pd = _toy_data()
    _, ax = plt.subplots()
    result = _toy_fn(
        X=pd.x,
        Y=pd.y,
        input_names=pd.input_names,
        output_name=pd.output_name,
        rescaler=IdentityRescaler(),
        ax=ax,
        threshold=0.42,
    )
    plt.close()
    assert isinstance(result, PlotFunctionResult)
    assert result.metadata["threshold"] == 0.42


def test_args_or_kwargs_rejected():
    def bad(X, ax, *args):
        pass

    with pytest.raises(TypeError, match=r"\*args"):
        panel_from(bad)

    def bad2(X, ax, **kw):
        pass

    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        panel_from(bad2)


def test_smooth_panel_2d_signature_matches_function_kwargs():
    import inspect

    sig = inspect.signature(SmoothPanel2D.__panel_fn__)
    fn_params = set(sig.parameters) - {"X", "Y", "input_names", "output_name", "ax"}
    panel_fields = set(SmoothPanel2D.model_fields)
    # `smooth_grid_params` is promoted to the `smooth_grid` CascadeLeaf field (bridged at draw)
    promoted = {"smooth_grid_params": "smooth_grid"}
    for p in fn_params:
        target = promoted.get(p, p)
        assert target in panel_fields, f"smooth_2d kwarg '{p}' missing from SmoothPanel2D"


def test_smooth_panel_2d_yaml_load():
    import dracon as dr

    pd = _toy_data()
    cfg = dr.loads(
        """
!SmoothPanel2D
plot_data: ${plot_data}
title: hello
draw_colorbar: false
""",
        enable_interpolation=True,
        context={**make_plot_context(), "plot_data": pd},
        raw_dict=True,
    )
    panel = cfg
    assert isinstance(panel, SmoothPanel2D)
    assert panel.title == "hello"
    assert panel.draw_colorbar is False


def test_txt_fn_renders_text():
    from jeanplot.panels.smooth_2d import SmoothPanel2D

    pd = _toy_data()
    panel = SmoothPanel2D(plot_data=pd, title="x", xlims=(0, 1), ylims=(0, 1))
    out = panel.render_txt()
    assert isinstance(out, str)
