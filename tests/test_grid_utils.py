"""Tests for grid_utils module"""

from jeanplot.core.grid_utils import CellRegion


class TestCellRegion:
    def test_empty_region(self):
        region = CellRegion(cells=[])
        assert region.bounds() == (0, 0, 0, 0)
        assert region.compute_boundary_edges({}) == []

    def test_single_cell_bounds(self):
        region = CellRegion(cells=[(0, 0)])
        assert region.bounds() == (0, 0, 0, 0)

    def test_multi_cell_bounds(self):
        region = CellRegion(cells=[(0, 0), (1, 2), (2, 1)])
        assert region.bounds() == (0, 2, 0, 2)

    def test_single_cell_boundary_edges(self):
        region = CellRegion(cells=[(0, 0)])
        cell_bounds = {(0, 0): (0, 0, 10, 10)}
        edges = region.compute_boundary_edges(cell_bounds)
        # single cell should have 4 boundary edges (top, right, bottom, left)
        assert len(edges) == 4
        # check all 4 sides are present
        expected = {
            ((0, 0), (10, 0)),   # top
            ((10, 0), (10, 10)), # right
            ((10, 10), (0, 10)), # bottom
            ((0, 10), (0, 0)),   # left
        }
        assert set(edges) == expected

    def test_horizontal_2_cells(self):
        region = CellRegion(cells=[(0, 0), (0, 1)])
        cell_bounds = {
            (0, 0): (0, 0, 10, 10),
            (0, 1): (12, 0, 22, 10),  # gap of 2
        }
        edges = region.compute_boundary_edges(cell_bounds)
        # should have: top (2), right (1), bottom (2), left (1), plus gap connectors (2)
        # top of (0,0): (0,0)-(10,0), top of (0,1): (12,0)-(22,0)
        # right of (0,1): (22,0)-(22,10)
        # bottom of (0,1): (22,10)-(12,10), bottom of (0,0): (10,10)-(0,10)
        # left of (0,0): (0,10)-(0,0)
        # gap connectors: (10,0)-(12,0) and (10,10)-(12,10)
        assert len(edges) == 8

    def test_vertical_2_cells(self):
        region = CellRegion(cells=[(0, 0), (1, 0)])
        cell_bounds = {
            (0, 0): (0, 0, 10, 10),
            (1, 0): (0, 12, 10, 22),  # gap of 2
        }
        edges = region.compute_boundary_edges(cell_bounds)
        # should have 8 edges similar to horizontal
        assert len(edges) == 8

    def test_l_shaped_region_has_concave_corners(self):
        # L-shaped region:
        # [X][ ]
        # [X][X]
        region = CellRegion(cells=[(0, 0), (1, 0), (1, 1)])
        cell_bounds = {
            (0, 0): (0, 0, 10, 10),
            (1, 0): (0, 12, 10, 22),
            (1, 1): (12, 12, 22, 22),
        }
        edges = region.compute_boundary_edges(cell_bounds)
        # L-shape should have concave corner edges
        # count should be > what a simple rectangle would have
        assert len(edges) > 8

    def test_offset_application(self):
        region = CellRegion(cells=[(0, 0)])
        cell_bounds = {(0, 0): (100, 100, 110, 110)}
        edges = region.compute_boundary_edges(cell_bounds, offset=(100, 100))
        # edges should be in local coords (0-10) not global (100-110)
        for (start, end) in edges:
            assert 0 <= start[0] <= 10
            assert 0 <= start[1] <= 10
            assert 0 <= end[0] <= 10
            assert 0 <= end[1] <= 10

    def test_missing_cell_bounds_handled(self):
        region = CellRegion(cells=[(0, 0), (0, 1)])
        cell_bounds = {(0, 0): (0, 0, 10, 10)}  # missing (0,1)
        edges = region.compute_boundary_edges(cell_bounds)
        # (0,0) has neighbor (0,1) in region, so right edge not drawn
        # only top, bottom, left edges from (0,0)
        assert len(edges) == 3

    def test_cell_set_property(self):
        region = CellRegion(cells=[(0, 0), (1, 1), (0, 0)])  # duplicate
        assert region.cell_set == {(0, 0), (1, 1)}
