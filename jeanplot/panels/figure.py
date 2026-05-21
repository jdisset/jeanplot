from pathlib import Path
from typing import Any
from pydantic import Field, PrivateAttr

from jeanplot.core.container import Container


class Figure(Container):
    """A Container that owns figure-level output, dpi, metadata, rc_context, theme.

    Has no orchestration of its own. `render()` forwards to `jeanplot.render(self)`.
    The renderer detects `Figure` and allocates a real `plt.Figure` plus one axes
    per drawable leaf `PlotPanel` from its laid-out bbox.
    """

    output_dir: str = "./"
    output_file: str | None = "unnamed.png"
    extra_output_paths: list[str] = Field(default_factory=list)
    dpi: int = 300
    rc_context: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    svg_id_prefix: str = "jp_"
    subtitle: str | None = None
    subtitle_kwargs: dict = Field(default_factory=dict)
    theme: Any | None = None

    _mpl_figure: Any = PrivateAttr(default=None)

    @property
    def output_path(self) -> Path | None:
        if not self.output_file:
            return None
        return Path(self.output_dir) / self.output_file

    def render(self, **kwargs):
        from jeanplot.render import render

        return render(self, **kwargs)


Figure.model_rebuild(force=True)
