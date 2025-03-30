from pydantic import Field, PrivateAttr
from .component import Component
import numpy as np
from .models import Size, LayoutConstraints


class Container(Component):
    """container that lays out and renders child components"""

    children: list[Component] = []
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)

    def measure(self, renderer=None) -> Size:
        """measure container based on children's bounds (excluding overlays)"""
        if not self.children:
            self._dimensions = self.min_dimensions
            self._transformed_aabb = self.compute_transformed_aabb()
            return self._dimensions

        # only consider non-overlay children for sizing
        layout_children = [c for c in self.children if not c.is_overlay]
        if not layout_children:
            self._dimensions = self.min_dimensions
            self._transformed_aabb = self.compute_transformed_aabb()
            return self._dimensions

        is_row = self.layout.direction == "row"
        main = 0  # main axis
        cross = 0  # cross axis

        for child in layout_children:
            child_dim = child._dimensions if child._dimensions else child.min_dimensions
            child_aabb = (
                child._transformed_aabb if hasattr(child, "_transformed_aabb") else child_dim
            )

            w = child_aabb.width
            h = child_aabb.height

            if hasattr(child, "style") and hasattr(child.style, "margin"):
                w += child.style.margin_left + child.style.margin_right
                h += child.style.margin_top + child.style.margin_bottom

            if is_row:
                main += w
                cross = max(cross, h)
            else:
                main += h
                cross = max(cross, w)

        if layout_children:
            main += self.layout.gap * (len(layout_children) - 1)

        width = main if is_row else cross
        height = cross if is_row else main

        # add insets and apply constraints
        insets = self.style.content_inset()
        width += insets[1] + insets[3]  # right + left
        height += insets[0] + insets[2]  # top + bottom

        self._dimensions = Size(
            width=min(max(self.min_dimensions.width, width), self.max_dimensions.width),
            height=min(max(self.min_dimensions.height, height), self.max_dimensions.height),
        )
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def measure_and_layout(self, renderer=None) -> Size:
        """process layout bottom-up"""
        layout_children = [c for c in self.children if not c.is_overlay]
        for child in layout_children:
            child.measure_and_layout(renderer)

        self.measure(renderer)

        self.apply_layout()

        overlay_children = [c for c in self.children if c.is_overlay]
        for overlay in overlay_children:
            overlay.measure(renderer)

        return self._dimensions

    def apply_layout(self):
        """position children based on layout constraints"""
        if not self.children:
            return

        # only layout non-overlay children
        layout_children = [c for c in self.children if not c.is_overlay]
        if not layout_children:
            return

        content_w, content_h = self.style.content_box(self._dimensions)
        insets = self.style.content_inset()
        content_x, content_y = insets[3], insets[0]  # left, top

        is_row = self.layout.direction == "row"
        self._layout_children(content_x, content_y, content_w, content_h, is_row, layout_children)

        stretched = self._apply_stretch_alignment(content_w, content_h)
        if stretched:
            self.measure()
            for container in stretched:
                if isinstance(container, Container) and container.children:
                    container.apply_layout()

    def _apply_stretch_alignment(self, content_w, content_h):
        """stretch children if needed"""
        layout_children = [c for c in self.children if not c.is_overlay]
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

            avail = (content_h - margins) if is_row else (content_w - margins)
            curr = child._dimensions.height if is_row else child._dimensions.width
            max_size = child.max_dimensions.height if is_row else child.max_dimensions.width

            if curr < avail and curr < max_size:
                new_size = min(avail, max_size)
                if curr != new_size:
                    if is_row:
                        child._dimensions.height = new_size
                    else:
                        child._dimensions.width = new_size
                    child._transformed_aabb = child.compute_transformed_aabb()
                    stretched.append(child)
        return stretched

    def _layout_children(self, content_x, content_y, content_w, content_h, is_row, layout_children):
        """position children in row or column layout (skip overlays)"""
        size_attr = "width" if is_row else "height"
        valid_children = [c for c in layout_children if getattr(c._transformed_aabb, size_attr) > 0]

        if not valid_children:
            return

        # calculate total size needed
        total = 0
        for child in valid_children:
            size = getattr(child._transformed_aabb, size_attr)
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )
                size += margins
            total += size

        gaps = self.layout.gap * (len(valid_children) - 1)
        required = total + gaps

        # calculate spacing
        avail = content_w if is_row else content_h
        extra = max(0, avail - required)
        pos = content_x if is_row else content_y

        # starting position
        if self.layout.justify_content == "center":
            pos += extra / 2
        elif self.layout.justify_content == "end":
            pos += extra

        # distributed spacing
        spacing = 0
        if len(valid_children) > 1:
            if self.layout.justify_content == "space-between":
                spacing = extra / (len(valid_children) - 1)
            elif self.layout.justify_content == "space-around":
                spacing = extra / len(valid_children)
                pos += spacing / 2
            elif self.layout.justify_content == "space-evenly":
                spacing = extra / (len(valid_children) + 1)
                pos += spacing

        curr = pos
        cross = content_y if is_row else content_x

        for child in valid_children:
            # position the child
            h_arg = content_h if is_row else None
            w_arg = None if is_row else content_w
            self._position_child(
                child, curr if is_row else cross, cross if is_row else curr, h_arg, w_arg
            )

            # advance position
            margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )

            size = getattr(child._transformed_aabb, size_attr)
            curr += size + margins + self.layout.gap + spacing

        # position zero-sized children at the start
        for child in self.children:
            if not child.is_overlay and getattr(child._transformed_aabb, size_attr) <= 0:
                self._position_child(
                    child,
                    pos if is_row else cross,
                    cross if is_row else pos,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )

    def _position_child(self, child, pos_x, pos_y, avail_h=None, avail_w=None):
        """position child with alignment"""
        x, y = pos_x, pos_y

        if hasattr(child, "style") and hasattr(child.style, "margin"):
            x += child.style.margin_left
            y += child.style.margin_top

        is_text = hasattr(child, "vertical_align") and hasattr(child, "align")
        is_row = self.layout.direction == "row"

        # vertical alignment for row layout
        if is_row and avail_h:
            margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = child.style.margin_top + child.style.margin_bottom
            eff_h = avail_h - margins
            child_h = child._transformed_aabb.height

            # center vertically
            if (is_text and child.vertical_align == "middle") or (
                not is_text and self.layout.align_items == "center"
            ):
                y = pos_y + (eff_h - child_h) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    y += child.style.margin_top
            # align to bottom
            elif (is_text and child.vertical_align == "bottom") or (
                not is_text and self.layout.align_items == "end"
            ):
                y = pos_y + eff_h - child_h
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    y -= child.style.margin_bottom

        # horizontal alignment for column layout
        if not is_row and avail_w:
            margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = child.style.margin_left + child.style.margin_right
            eff_w = avail_w - margins
            child_w = child._transformed_aabb.width

            # center horizontally
            if (is_text and child.align == "center") or (
                not is_text and self.layout.align_items == "center"
            ):
                x = pos_x + (eff_w - child_w) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    x += child.style.margin_left
            # align to right
            elif (is_text and child.align == "right") or (
                not is_text and self.layout.align_items == "end"
            ):
                x = pos_x + eff_w - child_w
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    x -= child.style.margin_right

        child.transform.translate = (x, y)

    def render(self, renderer, context, matrix: np.ndarray):
        """render container and children, with overlays on top"""
        # draw container background and border
        if self.style.background_color or self.style.border_color:
            renderer.render_rectangle(
                context,
                self._dimensions,
                self.style,
                matrix,
                component=self,
            )

        if self.debug:
            renderer.render_debug(context, self, matrix)

        regular_children = [c for c in self.children if not c.is_overlay]
        overlay_children = [c for c in self.children if c.is_overlay]

        # render regular children first
        for child in regular_children:
            child_matrix = child.compute_local_matrix()
            world_matrix = matrix @ child_matrix
            child.render(renderer, context, world_matrix)

        # render overlays on top
        for overlay in overlay_children:
            overlay_matrix = overlay.compute_local_matrix()
            world_matrix = matrix @ overlay_matrix
            overlay.render(renderer, context, world_matrix)

    def add_child(self, child: Component):
        """add child component"""
        self.children.append(child)
        child.parent = self
