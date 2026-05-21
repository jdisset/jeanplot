import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jeanplot
from jeanplot import Figure, PlotPanel, Size, LayoutConstraints
from jeanplot.data import PlotFunctionResult
from pydantic import PrivateAttr


class FakePanel(PlotPanel):
    _draw_calls: int = PrivateAttr(default=0)

    def draw(self, ax):
        self._draw_calls += 1
        ax.plot([0, 1], [0, 1])
        return PlotFunctionResult(rendering=None, metadata={"hello": "world"})


def _make_figure(tmpdir):
    fig = Figure(
        id="fig",
        output_dir=str(tmpdir),
        output_file="out.png",
        min_dimensions=Size(6.0, 3.0),
        layout=LayoutConstraints(direction="row"),
        dpi=50,
    )
    fig.add_child(FakePanel(id="p1", min_dimensions=Size(3.0, 3.0)))
    fig.add_child(FakePanel(id="p2", min_dimensions=Size(3.0, 3.0)))
    return fig


def test_figure_render_writes_file():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        mfig = jeanplot.render(fig)
        out = Path(td) / "out.png"
        assert out.exists()
        assert out.stat().st_size > 0
        for p in fig.children:
            assert p._draw_calls == 1
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_figure_self_render_equivalent():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        mfig = fig.render()
        out = Path(td) / "out.png"
        assert out.exists()
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_figure_metadata_collected_from_panels():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        jeanplot.render(fig)
        assert fig.metadata.get("hello") == "world"


def test_figure_output_path_override():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        target = Path(td) / "sub" / "elsewhere.png"
        jeanplot.render(fig, output_path=str(target))
        assert target.exists()


def test_figure_overwrite_false_skips_existing():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        out = Path(td) / "out.png"
        out.write_bytes(b"placeholder")
        result = jeanplot.render(fig, overwrite=False)
        assert result is None
        assert out.read_bytes() == b"placeholder"
