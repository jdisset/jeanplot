import tempfile
import matplotlib

matplotlib.use("Agg")

import jeanplot
from jeanplot import Figure, PlotPanel, Size, LayoutConstraints
from pydantic import PrivateAttr


class Recorder(PlotPanel):
    _bbox: tuple | None = PrivateAttr(default=None)

    def draw(self, ax):
        self._bbox = tuple(ax.get_position().bounds)


def test_row_layout_allocates_axes_at_expected_fractions():
    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="t.png",
            min_dimensions=Size(6.0, 2.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
        )
        panels = [
            Recorder(id=f"p{i}", min_dimensions=Size(2.0, 2.0),
                     axes_size=Size(2.0, 2.0))
            for i in range(3)
        ]
        for p in panels:
            fig.add_child(p)
        mfig = jeanplot.render(fig)
        for i, p in enumerate(panels):
            assert p._bbox is not None
            left, bottom, w, h = p._bbox
            assert abs(left - i / 3.0) < 1e-3
            assert abs(w - 1 / 3.0) < 1e-3
            assert abs(h - 1.0) < 1e-3
        import matplotlib.pyplot as plt

        plt.close(mfig)
