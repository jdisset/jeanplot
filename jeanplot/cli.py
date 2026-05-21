"""jeanplot CLI: render a Figure-typed YAML to an image file.

The program declares a typed `figure: Figure` field — the YAML composes via
`!Figure`, so parent-scope `!set_default` / `!require` propagate and every
flag the loaded YAML declares surfaces on the command line.
"""

from pathlib import Path
from typing import Annotated

from dracon.commandline import Arg, dracon_program
from pydantic import BaseModel, ConfigDict

from jeanplot import DEFAULT_TYPES, load_default_theme
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self) -> Figure:
        if self.output_dir is not None:
            self.figure.output_dir = str(self.output_dir)
        if self.output_file is not None:
            self.figure.output_file = self.output_file
        self.figure.render(overwrite=self.overwrite)
        return self.figure


if __name__ == "__main__":
    PlotJob.cli()
