"""Grid utilities for computing cell regions and boundaries"""

from pydantic import BaseModel


class CellRegion(BaseModel):
    cells: list[tuple[int, int]]

    @property
    def cell_set(self) -> set[tuple[int, int]]:
        return set(self.cells)

    def bounds(self) -> tuple[int, int, int, int]:
        if not self.cells:
            return (0, 0, 0, 0)
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (min(rows), max(rows), min(cols), max(cols))

    def compute_boundary_edges(
        self,
        cell_bounds: dict[tuple[int, int], tuple[float, float, float, float]],
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Compute boundary edges for this cell region, including gap connectors
        and concave corners for L-shaped regions.

        Args:
            cell_bounds: Dict mapping (row, col) -> (x0, y0, x1, y1) in global coords
            offset: Offset to subtract from global coords to get local coords

        Returns:
            List of ((x0, y0), (x1, y1)) edge segments
        """
        cells = self.cell_set
        if not cells:
            return []

        off_x, off_y = offset
        edges = []

        def get_bounds(r, c):
            if (r, c) not in cell_bounds:
                return None
            gx0, gy0, gx1, gy1 = cell_bounds[(r, c)]
            return (gx0 - off_x, gy0 - off_y, gx1 - off_x, gy1 - off_y)

        for r, c in cells:
            b = get_bounds(r, c)
            if not b:
                continue
            x0, y0, x1, y1 = b

            # boundary edges for exposed sides
            for dr, dc, edge in [
                (-1, 0, ((x0, y0), (x1, y0))),  # top
                (0, 1, ((x1, y0), (x1, y1))),  # right
                (1, 0, ((x1, y1), (x0, y1))),  # bottom
                (0, -1, ((x0, y1), (x0, y0))),  # left
            ]:
                if (r + dr, c + dc) not in cells:
                    edges.append(edge)

            # gap connectors to adjacent cells
            for dr, dc, cond1, cond2, corner_fn in [
                (
                    1,
                    0,
                    (0, -1),
                    (0, 1),
                    lambda b2: (((x0, y1), (b2[0], b2[1])), ((x1, y1), (b2[2], b2[1]))),
                ),
                (
                    0,
                    1,
                    (-1, 0),
                    (1, 0),
                    lambda b2: (((x1, y0), (b2[0], b2[1])), ((x1, y1), (b2[0], b2[3]))),
                ),
            ]:
                adj = get_bounds(r + dr, c + dc)
                if (r + dr, c + dc) in cells and adj:
                    e1, e2 = corner_fn(adj)
                    if (r + cond1[0], c + cond1[1]) not in cells:
                        edges.append(e1)
                    if (r + cond2[0], c + cond2[1]) not in cells:
                        edges.append(e2)

            # concave corners (L-shape)
            for dr, dc, corner_fn in [
                (1, 1, lambda b1, b2: ((b1[2], b1[1]), (b1[2], b2[3]), (b2[0], b2[3]))),
                (1, -1, lambda b1, b2: ((b1[0], b1[1]), (b1[0], b2[3]), (b2[2], b2[3]))),
                (-1, 1, lambda b1, b2: ((b1[2], b1[3]), (b1[2], b2[1]), (b2[0], b2[1]))),
                (-1, -1, lambda b1, b2: ((b1[0], b1[3]), (b1[0], b2[1]), (b2[2], b2[1]))),
            ]:
                adj1, adj2, diag = (r + dr, c), (r, c + dc), (r + dr, c + dc)
                if adj1 in cells and adj2 in cells and diag not in cells:
                    b1, b2 = get_bounds(*adj1), get_bounds(*adj2)
                    if b1 and b2:
                        p1, corner, p2 = corner_fn(b1, b2)
                        edges.extend([(p1, corner), (corner, p2)])

        return edges
