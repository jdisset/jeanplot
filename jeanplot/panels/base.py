from typing import Any
from pydantic import Field, PrivateAttr, model_validator
import matplotlib.axes

from jeanplot.core.container import Container
from jeanplot.core.models import BoxInset, Size
from jeanplot.data import PlotData, LazyPlotData, PlotFunctionResult


TITLE_ROOM = 0.3  # min top inset auto-reserved for ax.set_title()


class PlotPanel(Container):
    """Container whose content is a matplotlib Axes.

    `axes_size` sets the min data-area size; `style.padding` is the inset between
    the panel bbox and the axes. Everything else (margin, dimensions, border,
    layout, offset) behaves like any Container. The axes fills `bbox - padding`;
    `min_dimensions` derives from `axes_size + padding` unless set explicitly."""

    plot_data: PlotData | LazyPlotData | None = None
    rescaler: Any | None = None
    title: str | None = None
    title_kwargs: dict = Field(default_factory=dict)
    # Draw the title inside the axes (e.g. a z-slice label) instead of above it, so
    # it reserves no top room. The draw still positions it via title_kwargs (y/loc).
    title_inside: bool = False
    xtitle: str | None = None
    ytitle: str | None = None
    vtitle: str | None = None
    is_drawable: bool = True

    axes_size: Size = Field(default_factory=lambda: Size(width=2.5, height=2.0))

    _axes: matplotlib.axes.Axes | None = PrivateAttr(default=None)
    _mappable: Any | None = PrivateAttr(default=None)
    _last_metadata: dict = PrivateAttr(default_factory=dict)

    def _right_overflow(self) -> float:
        """Inches of content drawn to the RIGHT of the axes box (e.g. an out-of-axes
        colorbar). 0 here; panels that draw such content override and it is reserved
        as right inset by `effective_padding`, so the layout accounts for it."""
        return 0.0

    @property
    def effective_padding(self) -> BoxInset:
        """style.padding, with top bumped to TITLE_ROOM when there's a title above and
        the right inset grown to reserve any `_right_overflow` (e.g. a colorbar)."""
        p = self.safe_style.padding
        top = TITLE_ROOM if (self.title and not self.title_inside and p.top < TITLE_ROOM) else p.top
        right = p.right + self._right_overflow()
        if top == p.top and right == p.right:
            return p
        return BoxInset(top=top, right=right, bottom=p.bottom, left=p.left)

    @model_validator(mode="after")
    def _compute_min_dimensions(self):
        if "min_dimensions" in self._user_set_fields:
            return self
        p = self.effective_padding
        object.__setattr__(
            self,
            "min_dimensions",
            Size(
                width=self.axes_size.width + p.left + p.right,
                height=self.axes_size.height + p.top + p.bottom,
            ),
        )
        return self

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
