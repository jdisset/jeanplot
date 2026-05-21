import tempfile
import numpy as np
import matplotlib

matplotlib.use("Agg")

import jeanplot
from jeanplot import Figure, PlotPanel, Colorbar, Size, LayoutConstraints
from jeanplot.data import PlotFunctionResult
from pydantic import PrivateAttr


class Heatmappy(PlotPanel):
    def draw(self, ax):
        im = ax.imshow(np.arange(16).reshape(4, 4))
        return PlotFunctionResult(rendering=None, mappable=im)


class CallableColorbar(Colorbar):
    _called_with: object = PrivateAttr(default=None)
    _parent_mappable_seen: object = PrivateAttr(default=None)

    def draw(self, ax):
        self._called_with = ax
        self._parent_mappable_seen = getattr(self.parent, "_mappable", None)
        super().draw(ax)


def test_colorbar_overlay_receives_parent_axes_and_mappable():
    with tempfile.TemporaryDirectory() as td:
        fig = Figure(
            id="fig",
            output_dir=td,
            output_file="cb.png",
            min_dimensions=Size(4.0, 4.0),
            layout=LayoutConstraints(direction="row"),
            dpi=50,
        )
        heat = Heatmappy(id="h", min_dimensions=Size(4.0, 4.0))
        cbar = CallableColorbar(id="cb")
        heat.add_child(cbar)
        fig.add_child(heat)
        mfig = jeanplot.render(fig)
        assert cbar._called_with is heat._axes
        assert cbar._parent_mappable_seen is not None
        import matplotlib.pyplot as plt

        plt.close(mfig)
