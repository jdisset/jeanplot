"""Cascade-selectable smoothing config. `SmoothKernel` (per-point splat kernel) nests in
`SmoothGrid` (grid sampling); both are reached by a bare `SmoothKernel:` / `SmoothGrid:`
rule. `.params` feeds the dict-driven `smooth_*` plot fns (kernel under `smooth_params`).

Fields default to None so an un-styled panel yields `.params == {}` -- the plot fn then
keeps its own defaults (and the legacy `smooth_grid_params or knn_grid_params` fallback
still fires). The cascade fills what a theme sets; `.params` drops the rest."""

from typing import Literal

from pydantic import Field

from jeanplot.core.models import CascadeLeaf

RebalanceMode = Literal["smooth", "hard"]


class SmoothKernel(CascadeLeaf):
    min_points: int | None = None
    radius: float | None = None
    sigma_in_radius: float | None = None
    rebalance_values: float | None = None
    rebalance_values_mode: RebalanceMode | None = None
    rebalance_centroids: float | None = None
    rebalance_centroids_mode: RebalanceMode | None = None


class SmoothGrid(CascadeLeaf):
    grid_resolution: int | None = None
    max_centroid_offset_frac: float | None = None
    smooth_params: SmoothKernel = Field(default_factory=SmoothKernel)
