"""CascadeLeaf promotion: smoothing config is reached by bare `SmoothGrid:`/`SmoothKernel:`
rules, deep-merges across specificity, and bridges to the dict-driven plot fns via `.params`."""

import numpy as np
import pytest

from jeanplot import SmoothPanel1D, SmoothPanel2D, jstyle
from jeanplot.data import PlotData
from jeanplot.panels.smooth_3d import CubeStackPanel
from jeanplot.panels.smooth_spec import SmoothGrid, SmoothKernel


@pytest.fixture
def rules():
    jstyle.update(
        {
            "SmoothKernel": {"min_points": 5, "rebalance_centroids": 1, "rebalance_values": 0.8},
            "SmoothGrid": {"grid_resolution": 250, "max_centroid_offset_frac": 10},
            "SmoothPanel3D": {"SmoothGrid": {"grid_resolution": 180}},
            "CubeStackPanel": {"SmoothGrid": {"grid_resolution": 100}},
        }
    )
    yield
    jstyle.clear()


def _pd():
    return PlotData(
        xval=np.random.rand(50), yval=np.random.rand(50), input_names=["a", "b"], output_name="o"
    )


def test_bare_grid_and_kernel_reach_2d_leaf(rules):
    p = SmoothPanel2D()
    jstyle.apply(p)
    params = p.smooth_grid.params
    assert params["grid_resolution"] == 250
    assert params["max_centroid_offset_frac"] == 10.0
    # nested kernel leaf, keyed `smooth_params` for the consuming fn
    assert params["smooth_params"]["min_points"] == 5
    assert params["smooth_params"]["rebalance_values"] == 0.8
    # radius wasn't set by the rule -> dropped (fn keeps its own default)
    assert "radius" not in params["smooth_params"]


def test_kernel_shared_cross_peer_with_1d(rules):
    p1 = SmoothPanel1D()
    jstyle.apply(p1)
    assert p1.smooth.params["min_points"] == 5
    assert p1.smooth.params["rebalance_values"] == 0.8


def test_cube_face_specialization_deep_merges(rules):
    cube = CubeStackPanel(plot_data=_pd())
    sgp = cube._face_smooth_grid_params()
    assert sgp["grid_resolution"] == 100  # CubeStackPanel SmoothGrid specialization
    assert sgp["smooth_params"]["min_points"] == 5  # kernel inherited from bare rule


def test_user_set_leaf_field_wins_over_cascade(rules):
    p = SmoothPanel2D(smooth_grid=SmoothGrid(grid_resolution=42))
    jstyle.apply(p)
    assert p.smooth_grid.params["grid_resolution"] == 42


def test_leaf_params_drops_none_and_excludes_identity():
    # un-styled leaf -> empty params (fn keeps its own defaults); identity never leaks
    assert SmoothKernel(id="x", style_class=["c"]).params == {}
    assert SmoothKernel(min_points=7).params == {"min_points": 7}
