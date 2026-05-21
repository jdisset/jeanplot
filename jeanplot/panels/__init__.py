from jeanplot.panels.base import PlotPanel, Colorbar
from jeanplot.panels.figure import Figure
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import (
    SmoothPanel2D,
    SmoothGradMagnitudePanel2D,
    GradientFieldPanel2D,
)
from jeanplot.panels.smooth_3d import SmoothPanel3D, CubeView
from jeanplot.panels.mvp import MVPPanel
from jeanplot.panels.density import DensityPanel1D
from jeanplot.panels.scatter import GridHistogramPanel, ScatterPanel3D
from jeanplot.panels.violin import ViolinPanel
from jeanplot.panels.particle import ParticlePanel
from jeanplot.panels.stacked_poly import StackedPolyPanel
from jeanplot.panels.ascii_heatmap import AsciiHeatmapPanel
from jeanplot.panels.overlays import (
    IdentityLineOverlay,
    DiagonalPathOverlay,
    SliceOverlay,
    SliceChordOverlay,
    AdditionVsRemovalOverlay,
    DensityContourOverlay,
)
from jeanplot.panels.auto import auto_panel

__all__ = [
    "PlotPanel",
    "Colorbar",
    "Figure",
    "SmoothPanel1D",
    "SmoothPanel2D",
    "SmoothGradMagnitudePanel2D",
    "GradientFieldPanel2D",
    "SmoothPanel3D",
    "CubeView",
    "MVPPanel",
    "DensityPanel1D",
    "GridHistogramPanel",
    "ScatterPanel3D",
    "ViolinPanel",
    "ParticlePanel",
    "StackedPolyPanel",
    "AsciiHeatmapPanel",
    "IdentityLineOverlay",
    "DiagonalPathOverlay",
    "SliceOverlay",
    "SliceChordOverlay",
    "AdditionVsRemovalOverlay",
    "DensityContourOverlay",
    "auto_panel",
]
