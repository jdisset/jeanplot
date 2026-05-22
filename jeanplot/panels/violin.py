"""ViolinPanel — generated via `panel_from`."""

from jeanplot.panels.from_function import panel_from
from jeanplot.plots.violin import smooth_voxel_conditioned_violin

ViolinPanel = panel_from(smooth_voxel_conditioned_violin, name="ViolinPanel")
