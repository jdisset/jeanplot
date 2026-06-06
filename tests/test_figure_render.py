import tempfile
from pathlib import Path

import matplotlib
import pytest

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


class LabeledPanel(PlotPanel):
    def draw(self, ax):
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("xlabel")
        ax.set_ylabel("ylabel")


def test_axes_bbox_inset_by_style_padding():
    with tempfile.TemporaryDirectory() as td:
        panel = LabeledPanel(axes_size=Size(2.5, 2.0))
        panel.style.padding = {"left": 0.5, "right": 0.6, "bottom": 0.5}
        fig = Figure(panel, output_dir=str(td), output_file="out.png", dpi=50)
        mfig = fig.render()
        figw, figh = mfig.get_size_inches()
        bb = panel._axes.get_position()
        assert bb.x0 == pytest.approx(0.5 / figw, abs=1e-3)
        assert bb.y0 == pytest.approx(0.5 / figh, abs=1e-3)
        assert (bb.x1 - bb.x0) * figw == pytest.approx(2.5, abs=1e-3)
        assert (bb.y1 - bb.y0) * figh == pytest.approx(2.0, abs=1e-3)
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_panel_at_figure_edge_leaves_padding_margin():
    with tempfile.TemporaryDirectory() as td:
        panel = LabeledPanel(axes_size=Size(2.5, 2.0))
        panel.style.padding = {"left": 0.5, "bottom": 0.5}
        fig = Figure(panel, output_dir=str(td), output_file="out.png", dpi=50)
        mfig = fig.render()
        figw, figh = mfig.get_size_inches()
        bb = panel._axes.get_position()
        assert bb.x0 * figw == pytest.approx(0.5, abs=1e-3)
        assert bb.y0 * figh == pytest.approx(0.5, abs=1e-3)
        import matplotlib.pyplot as plt

        plt.close(mfig)


def test_figure_overwrite_false_skips_existing():
    with tempfile.TemporaryDirectory() as td:
        fig = _make_figure(td)
        out = Path(td) / "out.png"
        out.write_bytes(b"placeholder")
        result = jeanplot.render(fig, overwrite=False)
        assert result is None
        assert out.read_bytes() == b"placeholder"


def test_table_grid_collapsed_no_double_draw():
    """A Table's frame + separators are one collapsed grid, drawn once per line. The old
    model drew every shared edge twice (cell-right + next-cell-left, row-bottom +
    next-row-top); this asserts the new model emits each line exactly once."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_hex
    from jeanplot._figure_render import render_figure
    from jeanplot.core.table import GridStyle, LineStyle, Table, TableCell

    def cell(label):
        return TableCell(children=[FakePanel(min_dimensions=Size(1.6, 0.9))])

    with tempfile.TemporaryDirectory() as td:
        fig = Figure(output_dir=td, output_file="out.png", dpi=50)
        fig.add_child(
            Table(
                header_rows=1,
                data=[[cell("h0"), cell("h1")], [cell("a"), cell("b")], [cell("c"), cell("d")]],
                grid=GridStyle(
                    corner_radius=0.2,
                    frame=LineStyle(color="#aa0000", width=1.6),
                    header=LineStyle(color="#00aa00", width=1.0),
                    inner=LineStyle(color="#0000aa", width=0.5),
                ),
            )
        )
        mfig = render_figure(fig)

        def lines(hexcol):
            return [
                a
                for a in mfig.artists
                if isinstance(a, mlines.Line2D) and to_hex(a.get_color()) == hexcol
            ]

        # frame: rounded -> exactly one PathPatch (no per-side frame lines), drawn in the
        # same transFigure space as the inner lines so they can't drift apart
        frame = [
            a
            for a in mfig.artists
            if isinstance(a, mpatches.PathPatch) and to_hex(a.get_edgecolor()) == "#aa0000"
        ]
        assert len(frame) == 1
        assert frame[0].get_transform() is mfig.transFigure
        # header separator: exactly one line (below the single header row), not doubled
        assert len(lines("#00aa00")) == 1
        # interior: 3 column dividers (one per row, 2 cols) + 1 row divider (between the
        # two body rows; the header/body boundary is the header line) = 4, each drawn once
        assert len(lines("#0000aa")) == 4


def test_figure_render_draws_container_chrome():
    """A plain (non-Table) container's per-side border renders in the figure path, which
    otherwise only draws PlotPanel leaves."""
    import matplotlib.lines as mlines
    from jeanplot import Container
    from jeanplot._figure_render import render_figure
    from jeanplot.core.models import BoxStyle

    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            output_dir=td,
            output_file="out.png",
            min_dimensions=Size(6.0, 3.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
        )
        framed = Container(
            min_dimensions=Size(3.0, 3.0),
            style=BoxStyle(border_color="#123456", border_width=1.0),
            children=[FakePanel(min_dimensions=Size(2.5, 2.5))],
        )
        fig.add_child(framed)
        fig.add_child(FakePanel(min_dimensions=Size(3.0, 3.0)))
        mfig = render_figure(fig)

        borders = [
            a
            for a in mfig.artists
            if isinstance(a, mlines.Line2D) and str(a.get_color()).startswith("#123456")
        ]
        # one Line2D per drawn side of the bordered container (4 sides)
        assert len(borders) == 4


def test_figure_render_chrome_dashed_and_rounded():
    """corner_radius rounds a full box (FancyBboxPatch); border_style dashes the
    per-side grid lines."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from jeanplot import Container
    from jeanplot._figure_render import render_figure
    from jeanplot.core.models import BoxStyle
    from jeanplot.core.table import CellStyle

    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            output_dir=td,
            output_file="out.png",
            min_dimensions=Size(6.0, 3.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
        )
        rounded = Container(
            min_dimensions=Size(3.0, 3.0),
            style=BoxStyle(border_color="#abcdef", border_width=1.0, corner_radius=0.3),
            children=[FakePanel(min_dimensions=Size(2.5, 2.5))],
        )
        # a dashed single-side (bottom-only) border via CellStyle toggles
        dashed = Container(
            min_dimensions=Size(3.0, 3.0),
            style=CellStyle(
                border_color="#fedcba",
                border_width=1.0,
                border_style="dashed",
                border_top=False,
                border_left=False,
                border_right=False,
            ),
            children=[FakePanel(min_dimensions=Size(2.5, 2.5))],
        )
        fig.add_child(rounded)
        fig.add_child(dashed)
        mfig = render_figure(fig)
        from matplotlib.colors import to_hex

        # rounded full box -> exactly one FancyBboxPatch in that edge color
        fancy = [
            a
            for a in mfig.artists
            if isinstance(a, mpatches.FancyBboxPatch) and to_hex(a.get_edgecolor()) == "#abcdef"
        ]
        assert len(fancy) == 1
        # dashed one-side border -> a single non-solid Line2D
        dash_lines = [
            a
            for a in mfig.artists
            if isinstance(a, mlines.Line2D) and to_hex(a.get_color()) == "#fedcba"
        ]
        assert len(dash_lines) == 1
        assert dash_lines[0].get_linestyle() not in ("-", "solid")


def test_figure_theme_overrides_merge_at_render():
    """`Figure.theme_overrides` deep-merges onto `theme` at render (overrides win) —
    a real attribute to layer a jstyle subtree into, no sentinel `!define` var."""
    from jeanplot import Container
    from jeanplot._figure_render import render_figure
    from jeanplot.core.models import BoxStyle

    base = {
        "Container[style_class=framed]": {
            "style.border_color": "#111111",
            "style.border_width": 1.0,
        }
    }
    override = {"Container[style_class=framed]": {"style.border_color": "#22cc44"}}

    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            output_dir=td,
            output_file="out.png",
            min_dimensions=Size(4.0, 3.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
            theme=base,
            theme_overrides=override,
        )
        framed = Container(
            style_class=["framed"],
            style=BoxStyle(),
            min_dimensions=Size(3.0, 3.0),
            children=[FakePanel(min_dimensions=Size(2.0, 2.0))],
        )
        fig.add_child(framed)
        render_figure(fig)
        # override color wins over the base theme's color
        from matplotlib.colors import to_hex

        assert to_hex(framed.style.border_color) == "#22cc44"
