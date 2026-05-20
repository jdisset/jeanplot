from __future__ import annotations

from typing import Any

from jeanplot.core.component import AnchorComponent, Component
from jeanplot.core.path_utils import normalize_vector


def collect_anchors(component: Component) -> list[AnchorComponent]:
    """collect anchor components from child and anchor lists, deduped."""
    anchors = [a for a in getattr(component, "children", []) if isinstance(a, AnchorComponent)]
    anchors += [a for a in getattr(component, "anchor_points", []) if a not in anchors]

    valid_anchors = []
    for anchor in anchors:
        if not anchor.parent:
            anchor.parent = component
        valid_anchors.append(anchor)
    return valid_anchors


def get_effective_anchor_position(anchor: AnchorComponent) -> tuple[float, float] | None:
    """connection point extended by anchor direction/min_segment."""
    anchor_pos = anchor.get_world_origin()
    if anchor_pos is None:
        return None

    direction = getattr(anchor, "direction", None)
    min_len = getattr(anchor, "min_segment", 0.0)

    if direction and min_len > 1e-6:
        norm_dir = normalize_vector(direction, default=(0, 0))
        if norm_dir != (0, 0):
            return (
                anchor_pos[0] + norm_dir[0] * min_len,
                anchor_pos[1] + norm_dir[1] * min_len,
            )

    return anchor_pos


def find_best_anchor_pair(
    start_component: Component,
    end_component: Component,
) -> tuple[AnchorComponent, AnchorComponent] | None:
    """anchor pair with minimal effective-point distance."""
    start_anchors = collect_anchors(start_component)
    end_anchors = collect_anchors(end_component)
    if not start_anchors or not end_anchors:
        return None

    start_anchor_details: list[dict[str, Any]] = []
    for s_anchor in start_anchors:
        s_eff_pos = get_effective_anchor_position(s_anchor)
        if s_eff_pos:
            start_anchor_details.append({"anchor": s_anchor, "eff_pos": s_eff_pos})

    end_anchor_details: list[dict[str, Any]] = []
    for e_anchor in end_anchors:
        e_eff_pos = get_effective_anchor_position(e_anchor)
        if e_eff_pos:
            end_anchor_details.append({"anchor": e_anchor, "eff_pos": e_eff_pos})

    if not start_anchor_details or not end_anchor_details:
        return None

    best_pair = None
    min_dist_sq = float("inf")
    for s_detail in start_anchor_details:
        s_anchor = s_detail["anchor"]
        s_eff_pos = s_detail["eff_pos"]
        for e_detail in end_anchor_details:
            e_anchor = e_detail["anchor"]
            e_eff_pos = e_detail["eff_pos"]
            dist_sq = (s_eff_pos[0] - e_eff_pos[0]) ** 2 + (s_eff_pos[1] - e_eff_pos[1]) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_pair = (s_anchor, e_anchor)

    return best_pair
