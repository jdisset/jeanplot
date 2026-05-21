import matplotlib.axes
from pydantic import PrivateAttr

from jeanplot import PlotPanel, Size


class RecorderPanel(PlotPanel):
    _calls: list = PrivateAttr(default_factory=list)

    def draw(self, ax):
        self._calls.append(ax)


def test_plot_panel_draw_is_called_with_axes():
    import matplotlib.pyplot as plt

    panel = RecorderPanel(id="rec", min_dimensions=Size(1.0, 1.0))
    fig, ax = plt.subplots()
    panel.draw(ax)
    assert len(panel._calls) == 1
    assert isinstance(panel._calls[0], matplotlib.axes.Axes)


def test_plot_panel_render_txt_default_none():
    panel = RecorderPanel(id="rec", min_dimensions=Size(1.0, 1.0))
    assert panel.render_txt() is None


def test_non_drawable_panel_draw_returns_none():
    class Shell(PlotPanel):
        is_drawable: bool = False

    shell = Shell(id="shell", min_dimensions=Size(1.0, 1.0))
    assert shell.draw(None) is None
