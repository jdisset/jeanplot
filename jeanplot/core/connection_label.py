from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from pydantic import PrivateAttr

from jeanplot.core.connector import Connection
from jeanplot.core.debug import get_logger
from jeanplot.core.text import Text

logger = get_logger(__name__)


class ConnectionLabel(Text):
    """Text label positioned along a Connection's curve.

    Resolves a Connection by ID in the component tree, computes a point
    along its rendered path, and renders text at that location with
    optional tangent-aligned rotation and connection clipping.
    """

    connection: str = ""  # ID of target Connection
    distance: float = 0.5  # position along curve
    relative: bool = True  # True = 0-1 fraction, False = data units
    align_to_edge: bool = True  # rotate to follow edge tangent
    clip_connection: bool = False  # cut gap in connection line under text
    clip_padding: float = 1.0  # gap padding in data units
    deny_text: str | None = None  # case-insensitive regex; suppress label if text matches
    is_overlay: bool = True  # don't affect layout

    _resolved_connection: Connection | None = PrivateAttr(default=None)
    _clip_data: dict[str, Any] | None = PrivateAttr(default=None)
    _render_world_pos: tuple[float, float] | None = PrivateAttr(default=None)
    _render_angle_rad: float = PrivateAttr(default=0.0)
    _text_artist: Any | None = PrivateAttr(default=None)

    def _resolve_connection_ref(self) -> Connection | None:
        """walk ancestor tree to find Connection with matching id."""
        if self._resolved_connection is not None:
            return self._resolved_connection
        target_id = self.connection
        if not target_id:
            return None

        root = self
        while root.parent is not None:
            root = root.parent

        found = self._search_tree_for_id(root, target_id)
        if isinstance(found, Connection):
            self._resolved_connection = found
            return found
        return None

    @staticmethod
    def _search_tree_for_id(node: Any, target_id: str) -> Any | None:
        if getattr(node, "id", None) == target_id:
            return node
        for child in getattr(node, "children", []):
            result = ConnectionLabel._search_tree_for_id(child, target_id)
            if result is not None:
                return result
        for anchor in getattr(node, "anchor_points", []):
            result = ConnectionLabel._search_tree_for_id(anchor, target_id)
            if result is not None:
                return result
        return None

    def render(self, renderer: Any, context: Any, matrix: np.ndarray) -> None:
        if not self.show or not self.effective_text:
            return

        if self.deny_text and re.search(self.deny_text, self.effective_text, re.IGNORECASE):
            return

        conn = self._resolve_connection_ref()
        if conn is None:
            logger.warning(f"ConnectionLabel: could not resolve connection '{self.connection}'")
            return

        result = conn.get_point_along(self.distance, relative=self.relative)
        if result is None:
            return
        local_pt, _tangent = result

        conn_matrix = conn.compute_world_matrix()
        world_pt = conn_matrix @ np.array([local_pt[0], local_pt[1], 1.0])
        wx, wy = float(world_pt[0]), float(world_pt[1])
        self._render_world_pos = (wx, wy)

        # chord sampling gives a more stable tangent than direct evaluation
        angle_deg = 0.0
        angle_rad = 0.0
        if self.align_to_edge:
            chord_half = 15
            r0 = conn.get_point_along(self.distance - chord_half, relative=self.relative)
            r1 = conn.get_point_along(self.distance + chord_half, relative=self.relative)
            if r0 and r1:
                p0 = conn_matrix @ np.array([r0[0][0], r0[0][1], 1.0])
                p1 = conn_matrix @ np.array([r1[0][0], r1[0][1], 1.0])
                dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
            else:
                dx, dy = _tangent
            angle_deg = math.degrees(math.atan2(dy, dx))
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180
            angle_rad = math.radians(angle_deg)

        self._render_angle_rad = angle_rad
        renderer.render_connection_label(context, self, wx, wy, angle_deg, matrix)

    @staticmethod
    def finalize_clips(
        labels: list[ConnectionLabel],
        context: Any,
    ) -> None:
        """build compound clip paths cutting holes for each label under its connection."""
        import matplotlib.patches as mpatches
        from matplotlib.path import Path as MplPath

        fig = context.get_figure()
        if fig is None:
            return

        mpl_renderer = fig.canvas.get_renderer()
        inv_transform = context.transData.inverted()
        x0, x1 = context.get_xlim()
        y0, y1 = context.get_ylim()

        from collections import defaultdict

        groups: dict[str, list[ConnectionLabel]] = defaultdict(list)
        for label in labels:
            if label._text_artist is not None and label.connection:
                groups[label.connection].append(label)

        for conn_id, conn_labels in groups.items():
            target_gid = f"{conn_id}_main_curve"
            conn_patches = [p for p in context.patches if p.get_gid() == target_gid]
            if not conn_patches:
                continue

            # outer axes rect (CW) minus rotated text bbox holes (CCW)
            verts: list[tuple[float, float]] = [
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
                (x0, y0),
            ]
            codes: list[int] = [
                MplPath.MOVETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.CLOSEPOLY,
            ]

            for label in conn_labels:
                text_obj = label._text_artist
                if text_obj is None or label._render_world_pos is None:
                    continue

                cx, cy = label._render_world_pos
                angle_rad = label._render_angle_rad

                bbox_disp = text_obj.get_window_extent(mpl_renderer)
                bbox_data = inv_transform.transform_bbox(bbox_disp)
                bw, bh = bbox_data.width, bbox_data.height

                # back-compute unrotated text width/height from axis-aligned bbox
                ca, sa = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
                det = ca * ca - sa * sa
                if abs(det) > 1e-4:
                    tw = (bw * ca - bh * sa) / det
                    th = (bh * ca - bw * sa) / det
                else:
                    tw = th = (bw + bh) / (2 * math.sqrt(2))
                tw = max(tw, 0.0)
                th = max(th, 0.0)

                pad = label.clip_padding
                hw = tw / 2 + pad
                hh = th / 2 + pad

                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                corners = [
                    (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
                    for dx, dy in [(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)]
                ]
                verts.extend(corners + [corners[0]])
                codes.extend(
                    [
                        MplPath.MOVETO,
                        MplPath.LINETO,
                        MplPath.LINETO,
                        MplPath.LINETO,
                        MplPath.CLOSEPOLY,
                    ]
                )

            clip_path = MplPath(verts, codes)
            clip_patch = mpatches.PathPatch(clip_path, transform=context.transData, visible=False)
            context.add_patch(clip_patch)
            for p in conn_patches:
                p.set_clip_path(clip_patch)
