from pydantic import Field, model_validator
from .component import Component
import numpy as np
from .models import Size, LayoutConstraints
from .debug import debug_print


class Container(Component):
    """container that lays out and renders child components"""

    children: list[Component] = []
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)

    @model_validator(mode="after")
    def set_parent_for_children(self):
        for child in self.children:
            child.parent = self
        return self

    def _log_debug(self, message: str, data=None):
        debug_print(self.id or "Container", message, data)

    def measure(self, renderer=None) -> Size:
        """measure container based on children's bounds"""
        self._log_debug("Measuring container")
        for child in self.children:
            child.parent = self

        if not self.children:
            self._dimensions = self.min_dimensions
            self._transformed_aabb = self.compute_transformed_aabb()
            self._log_debug("No children, using min dimensions", self._dimensions)
            return self._dimensions

        # only consider non-overlay children for sizing
        layout_children = [c for c in self.children if not c.is_overlay]
        if not layout_children:
            self._dimensions = self.min_dimensions
            self._transformed_aabb = self.compute_transformed_aabb()
            self._log_debug("No layout children, using min dimensions", self._dimensions)
            return self._dimensions

        is_row = self.layout.direction == "row"
        main = 0  # main axis size
        cross = 0  # cross axis size

        for child in layout_children:
            # use measured dims if available, else min dims
            child_dim = getattr(child, "_dimensions", child.min_dimensions)
            # use transformed aabb if available, else basic dims
            child_aabb = getattr(child, "_transformed_aabb", child_dim)

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
        self._dimensions = Size(
            width=min(max(self.min_dimensions.width, width), self.max_dimensions.width),
            height=min(max(self.min_dimensions.height, height), self.max_dimensions.height),
        )
        self._transformed_aabb = self.compute_transformed_aabb()

        self._log_debug(
            "Measured container dimensions",
            {"width": self._dimensions.width, "height": self._dimensions.height},
        )
        return self._dimensions

    def measure_and_layout(self, renderer=None) -> Size:
        """process layout bottom-up"""
        self._log_debug("Starting measure_and_layout")

        # first measure non-overlay children
        layout_children = [c for c in self.children if not c.is_overlay]
        for child in layout_children:
            child.measure_and_layout(renderer)
            self._log_debug(
                f"Child {child.id} measured",
                {"dims": (child._dimensions.width, child._dimensions.height)},
            )

        # then measure this container
        self.measure(renderer)

        # then apply layout to position children
        self.apply_layout()
        self._log_debug("Layout applied")

        # finally measure overlays (connections, etc.)
        for overlay in [c for c in self.children if c.is_overlay]:
            overlay.measure(renderer)
            self._log_debug(
                f"Overlay {overlay.id} measured",
                {
                    "dims": (overlay._dimensions.width, overlay._dimensions.height),
                    "pos": overlay.transform.translate,
                },
            )

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

        # check if stretch alignment needs a remeasure/relayout cycle
        stretched = self._apply_stretch_alignment(content_w, content_h)
        if stretched:
            # remeasure container based on stretched children
            self.measure()
            # potentially relayout nested containers
            for container in stretched:
                if isinstance(container, Container) and container.children:
                    container.apply_layout()

    def _apply_stretch_alignment(self, content_w, content_h):
        """stretch children if needed based on 'stretch' alignment"""
        layout_children = [c for c in self.children if not c.is_overlay]
        if not layout_children or self.layout.align_items != "stretch":
            return []  # no stretching needed

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

            # available and current cross axis sizes
            avail_cross = (content_h - margins) if is_row else (content_w - margins)
            current_cross = child._dimensions.height if is_row else child._dimensions.width
            max_cross = child.max_dimensions.height if is_row else child.max_dimensions.width

            if current_cross < avail_cross and current_cross < max_cross:
                # calculate new size, respecting max constraint
                new_size = min(avail_cross, max_cross)
                # apply if significantly different
                if abs(current_cross - new_size) > 1e-6:
                    if is_row:
                        child._dimensions.height = new_size
                    else:
                        child._dimensions.width = new_size
                    # update transformed aabb after size change
                    child._transformed_aabb = child.compute_transformed_aabb()
                    stretched.append(child)
        return stretched

    def _layout_children(self, content_x, content_y, content_w, content_h, is_row, layout_children):
        """position children in row or column layout"""
        size_attr = "width" if is_row else "height"
        # filter out zero-size children
        valid_children = [c for c in layout_children if getattr(c._transformed_aabb, size_attr) > 0]

        if not valid_children:  # handle case with only zero-sized children
            # position all layout children at start
            for child in layout_children:
                self._position_child(
                    child,
                    content_x,
                    content_y,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )
            return

        # calculate total size needed by valid children
        total_main, total_gap = 0, self.layout.gap * (len(valid_children) - 1)

        for child in valid_children:
            size = getattr(child._transformed_aabb, size_attr)
            # include margins
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )
                size += margins
            total_main += size

        required_space = total_main + total_gap
        available_space = content_w if is_row else content_h
        extra_space = max(0, available_space - required_space)

        # starting position calculation
        current_pos = content_x if is_row else content_y
        justify = self.layout.justify_content

        if justify == "center":
            current_pos += extra_space / 2
        elif justify == "end":
            current_pos += extra_space
        elif justify in ("space-between", "space-around", "space-evenly"):
            spacing = 0
            if justify == "space-between" and len(valid_children) > 1:
                spacing = extra_space / (len(valid_children) - 1)
            elif justify == "space-around" and len(valid_children) > 0:
                spacing = extra_space / len(valid_children)
                current_pos += spacing / 2  # half at start
            elif justify == "space-evenly" and len(valid_children) + 1 > 0:
                spacing = extra_space / (len(valid_children) + 1)
                current_pos += spacing  # full at start

            # position children with spacing
            cross_pos = content_y if is_row else content_x

            for i, child in enumerate(valid_children):
                self._position_child(
                    child,
                    current_pos if is_row else cross_pos,
                    cross_pos if is_row else current_pos,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )

                # advance position
                size = getattr(child._transformed_aabb, size_attr)
                margins = 0
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    margins = (
                        child.style.margin_left + child.style.margin_right
                        if is_row
                        else child.style.margin_top + child.style.margin_bottom
                    )

                current_pos += size + margins + self.layout.gap
                if i < len(valid_children) - 1:
                    current_pos += spacing

            # position any remaining zero-sized children at start
            start_pos = content_x if is_row else content_y
            if justify == "center":
                start_pos += extra_space / 2
            elif justify == "end":
                start_pos += extra_space
            elif justify == "space-around":
                start_pos += spacing / 2 if len(valid_children) > 0 else 0
            elif justify == "space-evenly":
                start_pos += spacing if len(valid_children) + 1 > 0 else 0

            for child in layout_children:
                if getattr(child._transformed_aabb, size_attr) <= 0:
                    self._position_child(
                        child,
                        start_pos if is_row else cross_pos,
                        cross_pos if is_row else start_pos,
                        content_h if is_row else None,
                        None if is_row else content_w,
                    )

            return  # early return since we already positioned all children

        # for start, center, end justifications
        cross_pos = content_y if is_row else content_x

        for child in valid_children:
            self._position_child(
                child,
                current_pos if is_row else cross_pos,
                cross_pos if is_row else current_pos,
                content_h if is_row else None,
                None if is_row else content_w,
            )

            # advance position
            size = getattr(child._transformed_aabb, size_attr)
            margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )

            current_pos += size + margins + self.layout.gap

        # position zero-sized children at start
        start_pos = content_x if is_row else content_y
        if justify == "center":
            start_pos += extra_space / 2
        elif justify == "end":
            start_pos += extra_space

        for child in layout_children:
            if getattr(child._transformed_aabb, size_attr) <= 0:
                self._position_child(
                    child,
                    start_pos if is_row else cross_pos,
                    cross_pos if is_row else start_pos,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )

    def _position_child(self, child, pos_x, pos_y, avail_h=None, avail_w=None):
        """position child with alignment, respecting margins"""
        x, y = pos_x, pos_y

        # adjust for child's margins
        if hasattr(child, "style") and hasattr(child.style, "margin"):
            x += child.style.margin_left
            y += child.style.margin_top

        # cross-axis alignment
        is_text = hasattr(child, "vertical_align") and hasattr(child, "align")
        is_row = self.layout.direction == "row"

        # vertical alignment (cross axis for row layout)
        if is_row and avail_h is not None:
            child_margins_v = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                child_margins_v = child.style.margin_top + child.style.margin_bottom

            # effective available height after margins
            effective_h = avail_h - child_margins_v
            child_h = child._transformed_aabb.height

            # get alignment setting
            align = is_text and child.vertical_align or self.layout.align_items

            # apply alignment
            if align in ("center", "middle"):
                y = pos_y + (effective_h - child_h) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    y += child.style.margin_top
            elif align in ("end", "bottom"):
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    y = pos_y + avail_h - child.style.margin_bottom - child_h
                else:
                    y = pos_y + effective_h - child_h

        # horizontal alignment (cross axis for column layout)
        elif not is_row and avail_w is not None:
            child_margins_h = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                child_margins_h = child.style.margin_left + child.style.margin_right

            # effective available width after margins
            effective_w = avail_w - child_margins_h
            child_w = child._transformed_aabb.width

            # get alignment setting
            align = is_text and child.align or self.layout.align_items

            # apply alignment
            if align == "center":
                x = pos_x + (effective_w - child_w) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    x += child.style.margin_left
            elif align in ("end", "right"):
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    x = pos_x + avail_w - child.style.margin_right - child_w
                else:
                    x = pos_x + effective_w - child_w

        # set final translation
        child.transform.translate = (x, y)
        self._log_debug(f"Positioned child {child.id}", {"pos": (x, y)})

    def render(self, renderer, context, matrix: np.ndarray):
        """render container and children"""
        # draw container background and border
        if self.style.background_color or self.style.border_color:
            renderer.render_rectangle(context, self._dimensions, self.style, matrix, component=self)

        if self.debug:
            renderer.render_debug(context, self, matrix)
            self._log_debug("Rendering container", {"matrix": matrix.tolist()})

        # separate regular children and overlays
        regular = [c for c in self.children if not c.is_overlay]
        overlays = [c for c in self.children if c.is_overlay]

        # render regular children first
        for child in regular:
            child_matrix = matrix @ child.compute_local_matrix()
            if self.debug:
                self._log_debug(
                    f"Rendering regular child {child.id}", {"world_matrix": child_matrix.tolist()}
                )
            child.render(renderer, context, child_matrix)

        # render overlays on top
        for overlay in overlays:
            overlay_matrix = matrix @ overlay.compute_local_matrix()
            if self.debug:
                self._log_debug(
                    f"Rendering overlay {overlay.id}", {"world_matrix": overlay_matrix.tolist()}
                )
            overlay.render(renderer, context, overlay_matrix)

    def add_child(self, child: Component):
        """add child component and set parent link"""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            self._log_debug(f"Added child {child.id or 'Unnamed'}")

    def add_children(self, children: list[Component]):
        for child in children:
            self.add_child(child)
