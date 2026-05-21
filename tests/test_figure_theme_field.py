import tempfile
import matplotlib

matplotlib.use("Agg")

import jeanplot
from jeanplot import Figure, PlotPanel, Size, LayoutConstraints
from jeanplot.core.style import jstyle
from pydantic import PrivateAttr


class Recorder(PlotPanel):
    _draw_calls: int = PrivateAttr(default=0)

    def draw(self, ax):
        self._draw_calls += 1


def test_figure_theme_applied_during_render():
    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="t.png",
            min_dimensions=Size(2.0, 2.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
            theme={"[id=tinted]": {"title": "themed"}},
        )
        panel = Recorder(id="tinted", min_dimensions=Size(2.0, 2.0))
        fig.add_child(panel)
        mfig = jeanplot.render(fig)
        assert panel.title == "themed"
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_figure_no_theme_leaves_ambient_jstyle():
    with tempfile.TemporaryDirectory() as td:
        jstyle.clear()
        jstyle.update({"[id=x]": {"title": "ambient"}})
        ambient = jstyle._cascade
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="n.png",
            min_dimensions=Size(2.0, 2.0),
            dpi=50,
        )
        panel = Recorder(id="x", min_dimensions=Size(2.0, 2.0))
        fig.add_child(panel)
        mfig = jeanplot.render(fig)
        assert jstyle._cascade is ambient
        assert panel.title == "ambient"
        import matplotlib.pyplot as plt

        plt.close(mfig)
