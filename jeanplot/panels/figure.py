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
    # Extra extensions to ALSO save the figure as, colocated with `output_path` (same dir + stem,
    # swapped suffix). Derived at save time from the final path, so it survives any output_dir
    # override — unlike pre-built `extra_output_paths`. e.g. ["svg"] writes an svg twin of the pdf.
    also_save_exts: list[str] = Field(default_factory=list)
    dpi: int = 300
    rc_context: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    svg_id_prefix: str = "jp_"
    subtitle: str | None = None
    subtitle_kwargs: dict = Field(default_factory=dict)
    theme: Any | None = None
    # Per-figure theme deltas, deep-merged onto `theme` at render (overrides win, via
    # `merge_jstyle_rules`). A real attribute to layer a jstyle subtree into — no
    # sentinel `!define` var needed: `!MyFigure { theme_overrides: {Sel: {prop: v}} }`.
    theme_overrides: Any | None = None

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
