import numpy as np
import dracon as dr
import pytest

from jeanplot import PlotData, LazyPlotData, DataDimensions, make_context_from_types, DEFAULT_TYPES


def test_basic_plot_data_shapes():
    pd = PlotData(xval=np.linspace(0, 1, 10), yval=np.linspace(1, 2, 10), input_names=["x"])
    assert pd.x.shape == (10, 1)
    assert pd.y.shape == (10, 1)
    assert pd.dimensions == DataDimensions(input=1, output=1)


def test_2d_dimensions():
    pd = PlotData(
        xval=np.random.rand(50, 2),
        yval=np.random.rand(50),
        input_names=["a", "b"],
    )
    assert pd.dimensions == DataDimensions(input=2, output=1)


def test_3d_dimensions():
    pd = PlotData(
        xval=np.random.rand(50, 3),
        yval=np.random.rand(50),
        input_names=["a", "b", "c"],
    )
    assert pd.dimensions == DataDimensions(input=3, output=1)


def test_column_proteins_alias_reads_column_names():
    pd = PlotData(
        xval=np.zeros((4, 2)),
        yval=np.zeros((4, 1)),
        input_names=["x", "y"],
        column_names=["P1", "P2"],
    )
    assert pd.column_proteins == ["P1", "P2"]


def test_column_proteins_alias_writes_column_names():
    pd = PlotData(xval=np.zeros((4, 2)), yval=np.zeros((4, 1)), input_names=["x", "y"])
    pd.column_proteins = ["A", "B"]
    assert pd.column_names == ["A", "B"]


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        pd = PlotData(xval=np.zeros((10, 2)), yval=np.zeros((9, 1)), input_names=["x", "y"])
        pd.check_shapes()


def test_dracon_round_trip():
    ctx = make_context_from_types(DEFAULT_TYPES)
    yaml = """
!PlotData
xval: [[0.0, 0.1], [0.5, 0.6], [0.9, 1.0]]
yval: [0.1, 0.5, 0.9]
input_names: [a, b]
output_name: out
column_names: [P1, P2]
"""
    pd = dr.loads(yaml, context=ctx)
    assert isinstance(pd, PlotData)
    assert pd.x.shape == (3, 2)
    assert pd.y.shape == (3, 1)
    assert pd.column_names == ["P1", "P2"]


def test_lazy_plot_data_loads_on_access():
    def loader(_self):
        return np.linspace(0, 1, 8).reshape(-1, 1), np.linspace(2, 3, 8).reshape(-1, 1)

    pd = LazyPlotData(get_xy=loader, input_names=["x"])
    assert pd.xval is None
    assert pd.x.shape == (8, 1)
    assert pd.y.shape == (8, 1)
