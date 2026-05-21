from typing import Any
from pydantic import Field, PrivateAttr
import matplotlib.axes

from jeanplot.core.container import Container
from jeanplot.data import PlotData, LazyPlotData, PlotFunctionResult


class PlotPanel(Container):
    """Base for any component that claims a matplotlib Axes from its laid-out bbox.

    Subclasses implement `draw(self, ax)` and may override `render_txt()` for
    terminal output. Container layout, jstyle, and overlay mechanics are inherited.
    Numbers in min/max/dimensions are interpreted as inches by the figure renderer.
    """

    plot_data: PlotData | LazyPlotData | None = None
    rescaler: Any | None = None
    title: str | None = None
    title_kwargs: dict = Field(default_factory=dict)
    xtitle: str | None = None
    ytitle: str | None = None
    vtitle: str | None = None
    is_drawable: bool = True

    _axes: matplotlib.axes.Axes | None = PrivateAttr(default=None)
    _mappable: Any | None = PrivateAttr(default=None)
    _last_metadata: dict = PrivateAttr(default_factory=dict)

    def draw(self, ax: matplotlib.axes.Axes) -> PlotFunctionResult | None:
        if not self.is_drawable:
            return None
        raise NotImplementedError(f"{type(self).__name__} must implement draw()")

    def render_txt(self) -> str | None:
        return None


class Colorbar(PlotPanel):
    """Overlay panel that draws a colorbar against `parent._mappable`."""

    is_overlay: bool = True
    plot_data: None = None
    size: tuple[float, float] = (0.06, 0.85)
    position: tuple[float, float] = (1.05, 0.075)
    tick_props: dict = Field(default_factory=lambda: {"labelsize": 9, "pad": 3, "length": 4})
    label: str | None = None

    def draw(self, ax: matplotlib.axes.Axes) -> None:
        mappable = getattr(self.parent, "_mappable", None) if self.parent else None
        if mappable is None:
            return
        fig = ax.figure
        bbox = ax.get_position()
        x = bbox.x0 + self.position[0] * bbox.width
        y = bbox.y0 + self.position[1] * bbox.height
        w = self.size[0] * bbox.width
        h = self.size[1] * bbox.height
        cax = fig.add_axes((x, y, w, h))
        cb = fig.colorbar(mappable, cax=cax)
        if self.tick_props:
            cax.tick_params(**self.tick_props)
        if self.label:
            cb.set_label(self.label)
        self._axes = cax


PlotPanel.model_rebuild(force=True)
Colorbar.model_rebuild(force=True)
