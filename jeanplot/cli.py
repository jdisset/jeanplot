"""jeanplot CLI: render a Figure-typed YAML to an image file."""

import sys
from pathlib import Path
from typing import Annotated, Literal

from dracon.commandline import Arg, dracon_program
from pydantic import BaseModel, ConfigDict

from jeanplot import DEFAULT_TYPES, load_default_theme
from jeanplot._tui import RenderTUI, current_tui, use_tui
from jeanplot.compose import COMPOSE_HELPERS
from jeanplot.panels.figure import Figure

__all__ = ["PlotJob", "main", "load_default_theme"]


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
        Arg(short="v", help="Show component tree + full span tree."),
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
        tui = current_tui()
        if tui is not None:
            tui.configure(verbose=self.verbose, quiet=self.quiet, preview=self.preview)
        self.figure.render(overwrite=self.overwrite)
        return self.figure


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    quiet = any(a in ("-q", "--quiet") for a in args)
    with use_tui(RenderTUI(quiet=quiet)):
        PlotJob.main(argv)


if __name__ == "__main__":
    main()
