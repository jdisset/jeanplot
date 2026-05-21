"""AutoPanel — dim-dispatch helper picking SmoothPanel{1,2,3}D by data dim.

The YAML twin lives in `jeanplot/resources/templates/auto_panel.yaml` and uses
the dracon Constructor Slots pattern (`!fn` template + `!$(_PanelByDim[dim])`).
This Python helper is the code-side equivalent.
"""

from jeanplot.panels.base import PlotPanel
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.panels.smooth_3d import SmoothPanel3D

_PANEL_BY_DIM: dict[int, type[PlotPanel]] = {
    1: SmoothPanel1D,
    2: SmoothPanel2D,
    3: SmoothPanel3D,
}


def auto_panel(plot_data, *, force_dim: int | None = None, **kwargs) -> PlotPanel:
    """Pick the right SmoothPanel{1,2,3}D by data dim."""
    dim = force_dim if force_dim is not None else plot_data.dimensions.input
    if dim not in _PANEL_BY_DIM:
        raise ValueError(f"auto_panel: unsupported input dim={dim}")
    return _PANEL_BY_DIM[dim](plot_data=plot_data, **kwargs)
