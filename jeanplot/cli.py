"""jeanplot CLI: render a Figure-typed YAML to an image file.

The program declares a typed `figure: Figure` field — the YAML composes via
`!Figure`, so parent-scope `!set_default` / `!require` propagate and every
flag the loaded YAML declares surfaces on the command line.
"""

from pathlib import Path
from typing import Annotated, Literal

from dracon.commandline import Arg, dracon_program
from pydantic import BaseModel, ConfigDict

from jeanplot import DEFAULT_TYPES, load_default_theme
from jeanplot._tui import RenderTUI
from jeanplot.compose import COMPOSE_HELPERS
from jeanplot.panels.figure import Figure

__all__ = ["PlotJob", "load_default_theme"]


@dracon_program(
    name="jeanplot",
    description="Render a jeanplot Figure YAML to PNG/PDF/SVG.",
    context_types=DEFAULT_TYPES,
    context=COMPOSE_HELPERS,
)
class PlotJob(BaseModel):
    figure: Figure
    overwrite: bool = True

    output_dir: Annotated[
        Path | None,
        Arg(short="o", help="Override the figure's output_dir."),
    ] = None

    output_file: Annotated[
        str | None,
        Arg(help="Override the figure's output_file."),
    ] = None

    verbose: Annotated[
        bool,
        Arg(short="v", help="Show component tree + per-panel timings."),
    ] = False

    quiet: Annotated[
        bool,
        Arg(short="q", help="Suppress all terminal output."),
    ] = False

    preview: Annotated[
        Literal["auto", "on", "off"],
        Arg(help="Inline image preview in graphics-capable terminals."),
    ] = "auto"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self) -> Figure:
        if self.output_dir is not None:
            self.figure.output_dir = str(self.output_dir)
        if self.output_file is not None:
            self.figure.output_file = self.output_file
        tui = RenderTUI(verbose=self.verbose, quiet=self.quiet, preview=self.preview)
        self.figure.render(overwrite=self.overwrite, tui=tui)
        return self.figure


if __name__ == "__main__":
    PlotJob.cli()
