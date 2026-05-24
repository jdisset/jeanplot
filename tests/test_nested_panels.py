import tempfile
import matplotlib

matplotlib.use("Agg")

import jeanplot
from jeanplot import Figure, PlotPanel, Container, Size, LayoutConstraints
from pydantic import PrivateAttr


class Recorder(PlotPanel):
    _bbox: tuple | None = PrivateAttr(default=None)

    def draw(self, ax):
        self._bbox = tuple(ax.get_position().bounds)


def test_nested_containers_allocate_axes_at_correct_positions():
    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="t.png",
            min_dimensions=Size(4.0, 4.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
        )
        left = Container(
            id="left",
            min_dimensions=Size(2.0, 4.0),
            layout=LayoutConstraints(direction="column"),
        )
        right = Container(
            id="right",
            min_dimensions=Size(2.0, 4.0),
            layout=LayoutConstraints(direction="column"),
        )
        panels = []
        for parent_c, prefix in [(left, "L"), (right, "R")]:
            for j in range(2):
                p = Recorder(id=f"{prefix}{j}", min_dimensions=Size(2.0, 2.0),
                             axes_size=Size(2.0, 2.0))
                parent_c.add_child(p)
                panels.append(p)
        fig.add_child(left)
        fig.add_child(right)
        mfig = jeanplot.render(fig)

        bboxes = {p.id: p._bbox for p in panels}
        for k, b in bboxes.items():
            assert b is not None, k
        assert abs(bboxes["L0"][0] - 0.0) < 1e-3
        assert abs(bboxes["L1"][0] - 0.0) < 1e-3
        assert abs(bboxes["R0"][0] - 0.5) < 1e-3
        assert abs(bboxes["R1"][0] - 0.5) < 1e-3
        assert abs(bboxes["L0"][1] - 0.5) < 1e-3
        assert abs(bboxes["L1"][1] - 0.0) < 1e-3
        assert abs(bboxes["R0"][1] - 0.5) < 1e-3
        assert abs(bboxes["R1"][1] - 0.0) < 1e-3
        for b in bboxes.values():
            assert abs(b[2] - 0.5) < 1e-3
            assert abs(b[3] - 0.5) < 1e-3
        import matplotlib.pyplot as plt

        plt.close(mfig)
