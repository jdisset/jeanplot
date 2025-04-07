from pydantic import Field, BaseModel, PrivateAttr
from typing import List, Optional, Any
from .component import Component, AnchorComponent
import numpy as np
from .models import Size, LayoutConstraints, BoxStyle, Offset
from .debug import debug_print
from .style import jstyle


class Container(Component):
    """container that lays out and renders child components"""

    children: List[Component] = Field(default_factory=list)
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)
    anchor_points: List[Component] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """runs after model validation completes"""
        super().model_post_init(__context)
        self._setup_children_and_anchors()

    def _setup_children_and_anchors(self):
        """sets parent links and merges anchors into children"""
        children_to_add = []

        # set parent link for children
        for child in self.children:
            if child is not None and child.parent != self:
                child.parent = self

        # process anchor points
        processed_anchors = []
        if self.anchor_points:
            current_child_ids = {id(c) for c in self.children if c is not None}
            for anchor in self.anchor_points:
                if isinstance(anchor, Component):
                    anchor.parent = self
                    processed_anchors.append(anchor)
                    if id(anchor) not in current_child_ids:
                        children_to_add.append(anchor)

        # append missing anchors to children
        if children_to_add:
            new_children_list = getattr(self, "children", []) or []
            new_children_list.extend(children_to_add)
            object.__setattr__(self, "children", new_children_list)
            self._log_debug(f"added {len(children_to_add)} anchors")

        self.anchor_points = processed_anchors

    def _log_debug(self, message: str, data=None):
        if self.debug:
            debug_print(self.id or "Container", message, data)

    def _get_layout_children(self):
        """filter to get non-overlay, non-attached children"""
        return [
            c
            for c in self.children
            if c
            and not c.is_overlay
            and not getattr(c, "_resolved_attach_target", None)
            and not getattr(c, "attached_to", None)
        ]

    def measure(self, renderer=None) -> Size:
        """measure container based on children's bounds"""
        self._log_debug("measuring container")

        # ensure parent links
        for child in self.children:
            if child is not None and child.parent != self:
                child.parent = self

        layout_children = self._get_layout_children()

        if not layout_children:
            self._dimensions = self.min_dimensions.model_copy()
            self._transformed_aabb = self.compute_transformed_aabb()
            return self._dimensions

        is_row = self.layout.direction == "row"
        main, cross = 0, 0

        for child in layout_children:
            if not hasattr(child, "_dimensions") or child._dimensions.width == 0:
                child.measure_and_layout(renderer)

            child_aabb = child.compute_transformed_aabb()
            w, h = child_aabb.width, child_aabb.height

            if hasattr(child, "style") and hasattr(child.style, "margin"):
                w += child.style.margin_left + child.style.margin_right
                h += child.style.margin_top + child.style.margin_bottom

            if is_row:
                main += w
                cross = max(cross, h)
            else:
                main += h
                cross = max(cross, w)

        # add gap space
        if len(layout_children) > 1:
            main += self.layout.gap * (len(layout_children) - 1)

        # set width/height based on layout direction
        width, height = (main, cross) if is_row else (cross, main)

        # add insets and apply constraints
        insets = self.style.content_inset()
        width += insets[1] + insets[3]  # right + left
        height += insets[0] + insets[2]  # top + bottom

        # apply min/max constraints
        final_width = min(max(self.min_dimensions.width, width), self.max_dimensions.width)
        final_height = min(max(self.min_dimensions.height, height), self.max_dimensions.height)
        self._dimensions = Size(width=final_width, height=final_height)
        self._transformed_aabb = self.compute_transformed_aabb()

        return self._dimensions

    def measure_and_layout(self, renderer=None) -> Size:
        """process layout bottom-up, handling attached components"""
        self._log_debug(f"starting measure_and_layout")

        # apply styles
        jstyle.apply(self)

        # measure children first
        for child in self.children:
            if child is not None:
                if child.parent != self:
                    child.parent = self
                child.measure_and_layout(renderer)

        # measure container after children
        self.measure(renderer)

        # apply layout to position children
        self.apply_layout()

        # final AABB calculation
        self._transformed_aabb = self.compute_transformed_aabb()

        return self._dimensions

    def apply_layout(self):
        """position children based on layout constraints"""
        if not self.children:
            return

        layout_children = self._get_layout_children()
        if not layout_children:
            return

        content_w, content_h = self.style.content_box(self._dimensions)
        insets = self.style.content_inset()
        content_x, content_y = insets[3], insets[0]  # left, top

        # apply stretch if needed
        stretched = self._apply_stretch_alignment(content_w, content_h, layout_children)
        if stretched:
            self._log_debug(f"stretched {len(stretched)} children")
            self.measure()  # remeasure container
            content_w, content_h = self.style.content_box(self._dimensions)
            content_x, content_y = insets[3], insets[0]

        # position children
        is_row = self.layout.direction == "row"
        self._layout_children(content_x, content_y, content_w, content_h, is_row, layout_children)

    def _apply_stretch_alignment(self, content_w, content_h, layout_children):
        """stretch children if needed based on 'stretch' alignment"""
        if not layout_children or self.layout.align_items != "stretch":
            return []

        stretched = []
        is_row = self.layout.direction == "row"

        for child in layout_children:
            margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_top + child.style.margin_bottom
                    if is_row
                    else child.style.margin_left + child.style.margin_right
                )

            avail_cross = max(0, (content_h - margins) if is_row else (content_w - margins))
            current_cross = child._dimensions.height if is_row else child._dimensions.width
            max_cross = child.max_dimensions.height if is_row else child.max_dimensions.width
            min_cross = child.min_dimensions.height if is_row else child.min_dimensions.width

            target_cross = min(max(min_cross, avail_cross), max_cross)

            if abs(current_cross - target_cross) > 1e-6:
                if is_row:
                    child._dimensions.height = target_cross
                else:
                    child._dimensions.width = target_cross
                child._transformed_aabb = child.compute_transformed_aabb()
                stretched.append(child)

        return stretched

    def _get_margins(self, child, is_row):
        """get relevant margin values for layout calculations"""
        if hasattr(child, "style") and hasattr(child.style, "margin"):
            if is_row:
                return (
                    child.style.margin_left + child.style.margin_right,
                    child.style.margin_top + child.style.margin_bottom,
                    child.style.margin_left,
                    child.style.margin_top,
                )
            else:
                return (
                    child.style.margin_top + child.style.margin_bottom,
                    child.style.margin_left + child.style.margin_right,
                    child.style.margin_top,
                    child.style.margin_left,
                )
        return 0, 0, 0, 0

    def _layout_children(self, content_x, content_y, content_w, content_h, is_row, layout_children):
        """position children in row or column layout"""
        size_attr = "width" if is_row else "height"
        valid_children = [
            c for c in layout_children if getattr(c._transformed_aabb, size_attr, 0) > 0
        ]

        if not valid_children:
            for child in layout_children:
                self._position_child(
                    child,
                    content_x,
                    content_y,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )
            return

        # calculate spacing based on justification
        total_main = 0
        total_gap = self.layout.gap * (len(valid_children) - 1)

        for child in valid_children:
            size = getattr(child._transformed_aabb, size_attr)
            main_margins, _, _, _ = self._get_margins(child, is_row)
            total_main += size + main_margins

        required_space = total_main + total_gap
        available_space = content_w if is_row else content_h
        extra_space = max(0, available_space - required_space)

        # calculate starting position and spacing
        current_pos = content_x if is_row else content_y
        cross_pos = content_y if is_row else content_x
        justify = self.layout.justify_content

        spacing = 0
        if justify == "center":
            current_pos += extra_space / 2
        elif justify == "end":
            current_pos += extra_space
        elif justify == "space-between" and len(valid_children) > 1:
            spacing = extra_space / (len(valid_children) - 1)
        elif justify == "space-around" and len(valid_children) > 0:
            spacing = extra_space / len(valid_children)
            current_pos += spacing / 2
        elif justify == "space-evenly" and len(valid_children) > 0:
            spacing = extra_space / (len(valid_children) + 1)
            current_pos += spacing

        # position valid children
        for i, child in enumerate(valid_children):
            self._position_child(
                child,
                current_pos if is_row else cross_pos,
                cross_pos if is_row else current_pos,
                content_h if is_row else None,
                None if is_row else content_w,
            )

            size = getattr(child._transformed_aabb, size_attr)
            main_margins, _, _, _ = self._get_margins(child, is_row)
            current_pos += size + main_margins + self.layout.gap

            # apply spacing for justification if needed
            if (
                justify in ("space-between", "space-around", "space-evenly")
                and i < len(valid_children) - 1
            ):
                current_pos += spacing

        # handle zero-sized children
        start_pos = content_x if is_row else content_y
        if justify == "center":
            start_pos += extra_space / 2
        elif justify == "end":
            start_pos += extra_space
        elif justify == "space-around" and len(valid_children) > 0:
            start_pos += spacing / 2
        elif justify == "space-evenly" and len(valid_children) > 0:
            start_pos += spacing

        for child in layout_children:
            if getattr(child._transformed_aabb, size_attr, 0) <= 0:
                self._position_child(
                    child,
                    start_pos if is_row else cross_pos,
                    cross_pos if is_row else start_pos,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )

    def _position_child(self, child, flow_pos_x, flow_pos_y, avail_h=None, avail_w=None):
        """position child with proper alignment"""
        child_w, child_h = child._dimensions.width, child._dimensions.height
        parent_content_w, parent_content_h = self.style.content_box(self._dimensions)
        parent_dims = Size(width=parent_content_w, height=parent_content_h)
        parent_inset_l, parent_inset_t = (
            self.style.content_inset()[3],
            self.style.content_inset()[0],
        )

        # check if child has custom offset
        has_custom_offset = (
            child.offset.absolute != (0.0, 0.0)
            or child.offset.relative != (0.0, 0.0)
            or child.offset.parent_relative != (0.0, 0.0)
        )

        if has_custom_offset:
            # use child's specified offset
            child_base_x, child_base_y = child.offset.compute(child._dimensions, parent_dims)
            final_x = parent_inset_l + child_base_x
            final_y = parent_inset_t + child_base_y
        else:
            # position based on layout flow and alignment
            x, y = flow_pos_x, flow_pos_y

            # get margins
            margin_l, margin_r, margin_t, margin_b = 0, 0, 0, 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margin_t, margin_r, margin_b, margin_l = child.style.margin

            x += margin_l
            y += margin_t

            # apply cross-axis alignment
            is_row = self.layout.direction == "row"
            if is_row and avail_h is not None:
                content_h = max(0, avail_h - margin_t - margin_b)
                align = getattr(child, "vertical_align", self.layout.align_items)
                if align in ("center", "middle"):
                    y = flow_pos_y + margin_t + (content_h - child_h) / 2
                elif align in ("end", "bottom"):
                    y = flow_pos_y + avail_h - margin_b - child_h
            elif not is_row and avail_w is not None:
                content_w = max(0, avail_w - margin_l - margin_r)
                align = getattr(child, "align", self.layout.align_items)
                if align == "center":
                    x = flow_pos_x + margin_l + (content_w - child_w) / 2
                elif align in ("end", "right"):
                    x = flow_pos_x + avail_w - margin_r - child_w

            final_x, final_y = x, y

        # store final position in child's offset
        child.offset = Offset(absolute=(final_x, final_y))

    def render(self, renderer, context, matrix: np.ndarray):
        """render container and children"""
        if not self.show:
            return

        # render container background/border
        if self.style.background_color or (self.style.border_color and self.style.border_width > 0):
            renderer.render_rectangle(context, self._dimensions, self.style, matrix, component=self)

        # render debug if needed
        if self.debug:
            renderer.render_debug(context, self, matrix)

        # render children
        for child in self.children:
            if child is not None and child.show:
                child_matrix = child.compute_world_matrix(parent_matrix=matrix)
                child.render(renderer, context, child_matrix)

    def add_child(self, child: Component):
        """add child component and set parent link"""
        if self.children is None:
            self.children = []

        if not any(c is child for c in self.children):
            self.children.append(child)
            child.parent = self
        elif child.parent != self:
            child.parent = self

        # add to anchor points if applicable
        if isinstance(child, AnchorComponent):
            if self.anchor_points is None:
                self.anchor_points = []
            if not any(a is child for a in self.anchor_points):
                self.anchor_points.append(child)

    def add_children(self, children: list[Component]):
        for child in children:
            self.add_child(child)
