"""Overlay panels that draw on top of their parent panel's axes.

Each overlay is a PlotPanel with `is_overlay=True`. The renderer hands them the
same axes their parent uses; they read `self.parent._axes` / `self.parent._mappable`
as needed.
"""

from typing import Any, Literal

from pydantic import Field

from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel
from jeanplot.plots import overlays as _o


class IdentityLineOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    color: str = "grey"
    ls: str = "--"
    lw: float = 1.0
    alpha: float = 0.8

    def draw(self, ax) -> PlotFunctionResult | None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lo = max(x0, y0)
        hi = min(x1, y1)
        ax.plot([lo, hi], [lo, hi], color=self.color, ls=self.ls, lw=self.lw, alpha=self.alpha)
        return PlotFunctionResult(rendering=None, metadata={})


class DiagonalPathOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    t_raw_values: list[float]
    s_raw_range: tuple[float, float]
    angle_deg: float = 45.0
    n: int = 400
    colors: list[Any] | None = None
    line_props: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        if self.rescaler is None:
            return None
        _o.plot_diagonal_paths(
            ax,
            self.t_raw_values,
            self.s_raw_range,
            self.rescaler,
            colors=self.colors,
            n=self.n,
            angle_deg=self.angle_deg,
            line_props=self.line_props,
        )
        return PlotFunctionResult(rendering=None, metadata={})


class SliceOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    slice_axis: Literal["x", "y", "s", "t"]
    slice_values_raw: list[float]
    var_range_raw: tuple[float, float] | None = None
    n: int = 400
    colors: list[Any] | None = None
    line_props: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        if self.rescaler is None:
            return None
        _o.plot_slice_overlay(
            ax,
            self.slice_axis,
            self.slice_values_raw,
            self.rescaler,
            var_range_raw=self.var_range_raw,
            colors=self.colors,
            n=self.n,
            line_props=self.line_props,
        )
        return PlotFunctionResult(rendering=None, metadata={})


class SliceChordOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    X: Any
    Y: Any
    slices: Any
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    colors: list[Any] | None = None
    knn_stats_params: dict = Field(default_factory=dict)
    res: int = 100
    n_curve: int = 200
    chord_props: dict = Field(default_factory=dict)

    def draw(self, ax) -> PlotFunctionResult | None:
        _o.plot_slice_chords(
            ax,
            self.X,
            self.Y,
            self.slices,
            self.xlims,
            rescaler=self.rescaler,
            colors=self.colors,
            knn_stats_params=self.knn_stats_params,
            res=self.res,
            n_curve=self.n_curve,
            chord_props=self.chord_props,
        )
        return PlotFunctionResult(rendering=None, metadata={})


class AdditionVsRemovalOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    X_lat: Any
    Y_lat: Any
    slice_values_raw: list[float]
    anchor_raw_values: list[float]
    colors: list[Any] | None = None
    knn_stats_params: dict = Field(default_factory=dict)
    max_centroid_offset_frac: float = 0.0
    line_props: dict = Field(default_factory=dict)
    res: int = 200

    def draw(self, ax) -> PlotFunctionResult | None:
        if self.rescaler is None:
            return None
        _o.plot_addition_vs_removal_overlay(
            ax,
            self.X_lat,
            self.Y_lat,
            self.slice_values_raw,
            self.anchor_raw_values,
            self.rescaler,
            colors=self.colors,
            knn_stats_params=self.knn_stats_params,
            max_centroid_offset_frac=self.max_centroid_offset_frac,
            line_props=self.line_props,
            res=self.res,
        )
        return PlotFunctionResult(rendering=None, metadata={})


class LinearityReferenceOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    X: Any
    Y: Any
    slices: Any
    xlims: tuple[float | None, float | None] = (0.0, 1.0)
    colors: list[Any] | None = None
    knn_stats_params: dict = Field(default_factory=dict)
    head_frac: float = 0.1
    tail_frac: float = 0.1
    show_head: bool = True
    show_tail: bool = True
    show_chord: bool = True
    line_props: dict = Field(default_factory=dict)
    head_props: dict | None = None
    tail_props: dict | None = None
    chord_props: dict | None = None
    res: int = 200
    n_curve: int = 200

    def draw(self, ax) -> PlotFunctionResult | None:
        _o.plot_linearity_reference(
            ax,
            self.X,
            self.Y,
            self.slices,
            rescaler=self.rescaler,
            xlims=self.xlims,
            colors=self.colors,
            knn_stats_params=self.knn_stats_params,
            head_frac=self.head_frac,
            tail_frac=self.tail_frac,
            show_head=self.show_head,
            show_tail=self.show_tail,
            show_chord=self.show_chord,
            line_props=self.line_props,
            head_props=self.head_props,
            tail_props=self.tail_props,
            chord_props=self.chord_props,
            res=self.res,
            n_curve=self.n_curve,
        )
        if getattr(self.parent, "show_legend", False):
            ax.legend(**(getattr(self.parent, "legend_kwargs", None) or {}))
        return PlotFunctionResult(rendering=None, metadata={})


class DensityContourOverlay(PlotPanel):
    is_overlay: bool = True
    plot_data: None = None
    x: Any
    y: Any
    levels: int = 5
    color: str = "k"
    lw: float = 0.5
    alpha: float = 0.6

    def draw(self, ax) -> PlotFunctionResult | None:
        import numpy as np
        from scipy.stats import gaussian_kde

        x = np.asarray(self.x).ravel()
        y = np.asarray(self.y).ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 5:
            return None
        kde = gaussian_kde(np.vstack([x[finite], y[finite]]))
        xg = np.linspace(x[finite].min(), x[finite].max(), 80)
        yg = np.linspace(y[finite].min(), y[finite].max(), 80)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        ax.contour(
            XX, YY, ZZ, levels=self.levels, colors=self.color, linewidths=self.lw, alpha=self.alpha
        )
        return PlotFunctionResult(rendering=None, metadata={})


IdentityLineOverlay.model_rebuild(force=True)
DiagonalPathOverlay.model_rebuild(force=True)
SliceOverlay.model_rebuild(force=True)
SliceChordOverlay.model_rebuild(force=True)
AdditionVsRemovalOverlay.model_rebuild(force=True)
LinearityReferenceOverlay.model_rebuild(force=True)
DensityContourOverlay.model_rebuild(force=True)
