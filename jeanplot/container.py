# file: jeanplot/container.py
from pydantic import Field, PrivateAttr, model_validator
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
        """helper to log debug messages with component id"""
        debug_print(self.id or "Container", message, data)

    def measure(self, renderer=None) -> Size:
        """measure container based on children's bounds (excluding overlays)"""
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
        main = 0  # main axis
        cross = 0  # cross axis

        for child in layout_children:
            # use measured dims if available, else min dims
            child_dim = child._dimensions if hasattr(child, "_dimensions") else child.min_dimensions
            # use transformed aabb if available, else basic dims
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
        overlay_children = [c for c in self.children if c.is_overlay]
        for overlay in overlay_children:
            # overlays might depend on final positions, so measure them last
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
        # this is a simplified approach; complex cases might need more iterations
        stretched = self._apply_stretch_alignment(content_w, content_h)
        if stretched:
            # remeasure container based on stretched children
            self.measure()
            # potentially relayout nested containers if they were stretched
            for container in stretched:
                if isinstance(container, Container) and container.children:
                    container.apply_layout()  # relayout children within stretched container

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

            # available space in cross axis
            avail_cross_size = (content_h - margins) if is_row else (content_w - margins)
            # current size in cross axis
            current_cross_size = child._dimensions.height if is_row else child._dimensions.width
            # max size constraint in cross axis
            max_cross_size = child.max_dimensions.height if is_row else child.max_dimensions.width

            if current_cross_size < avail_cross_size and current_cross_size < max_cross_size:
                # calculate new size, respecting max constraint
                new_size = min(avail_cross_size, max_cross_size)
                # apply if different
                if abs(current_cross_size - new_size) > 1e-6:  # floating point comparison
                    if is_row:
                        child._dimensions.height = new_size
                    else:
                        child._dimensions.width = new_size
                    # update transformed aabb after size change
                    child._transformed_aabb = child.compute_transformed_aabb()
                    stretched.append(child)
        return stretched

    def _layout_children(self, content_x, content_y, content_w, content_h, is_row, layout_children):
        """position children in row or column layout (skip overlays)"""
        size_attr = "width" if is_row else "height"
        # filter out children with zero size in the main layout direction to avoid division by zero later
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

        # calculate total size needed by valid children along main axis
        total_main_axis_size = 0
        for child in valid_children:
            size = getattr(child._transformed_aabb, size_attr)
            # include margins in total size calculation
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )
                size += margins
            total_main_axis_size += size

        # total gap space
        total_gap_space = self.layout.gap * (len(valid_children) - 1)
        # total required space along main axis
        required_main_axis_space = total_main_axis_size + total_gap_space

        # available space along main axis
        available_main_axis_space = content_w if is_row else content_h
        # calculate extra space
        extra_space = max(0, available_main_axis_space - required_main_axis_space)

        # starting position along main axis
        current_pos = content_x if is_row else content_y
        if self.layout.justify_content == "center":
            current_pos += extra_space / 2
        elif self.layout.justify_content == "end":
            current_pos += extra_space

        # spacing between elements (for space-between, space-around, space-evenly)
        inter_element_spacing = 0
        num_gaps = len(valid_children) - 1
        num_spaces_around = len(valid_children)
        num_spaces_evenly = len(valid_children) + 1

        if num_gaps > 0 and self.layout.justify_content == "space-between":
            inter_element_spacing = extra_space / num_gaps
        elif num_spaces_around > 0 and self.layout.justify_content == "space-around":
            inter_element_spacing = extra_space / num_spaces_around
            current_pos += inter_element_spacing  # initial offset for space-around
        elif num_spaces_evenly > 0 and self.layout.justify_content == "space-evenly":
            inter_element_spacing = extra_space / num_spaces_evenly
            current_pos += inter_element_spacing  # initial offset for space-evenly

        # fixed position along cross axis
        cross_axis_pos = content_y if is_row else content_x

        # position valid children
        for i, child in enumerate(valid_children):
            # determine cross-axis arguments for _position_child
            cross_avail_h = content_h if is_row else None
            cross_avail_w = None if is_row else content_w

            # position the child
            self._position_child(
                child,
                current_pos if is_row else cross_axis_pos,  # main axis position
                cross_axis_pos if is_row else current_pos,  # cross axis position
                cross_avail_h,  # available height for vertical alignment (if row)
                cross_avail_w,  # available width for horizontal alignment (if col)
            )

            # advance position for next child
            child_margins = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                child_margins = (
                    child.style.margin_left + child.style.margin_right
                    if is_row
                    else child.style.margin_top + child.style.margin_bottom
                )

            child_size = getattr(child._transformed_aabb, size_attr)
            current_pos += child_size + child_margins

            # add gap and spacing
            if i < len(valid_children) - 1:  # not the last element
                current_pos += self.layout.gap
                if self.layout.justify_content in ["space-between", "space-around", "space-evenly"]:
                    current_pos += inter_element_spacing
            elif (
                self.layout.justify_content == "space-around"
            ):  # add trailing space for space-around
                current_pos += inter_element_spacing

        # position zero-sized children at the calculated start position
        start_pos = content_x if is_row else content_y
        if self.layout.justify_content == "center":
            start_pos += extra_space / 2
        elif self.layout.justify_content == "end":
            start_pos += extra_space
        elif self.layout.justify_content == "space-around":
            start_pos += inter_element_spacing if num_spaces_around > 0 else 0
        elif self.layout.justify_content == "space-evenly":
            start_pos += inter_element_spacing if num_spaces_evenly > 0 else 0

        for child in layout_children:
            if getattr(child._transformed_aabb, size_attr) <= 0:
                self._position_child(
                    child,
                    start_pos if is_row else cross_axis_pos,
                    cross_axis_pos if is_row else start_pos,
                    content_h if is_row else None,
                    None if is_row else content_w,
                )

    def _position_child(self, child, pos_x, pos_y, avail_h=None, avail_w=None):
        """position child with alignment, respecting margins"""
        # start with base position provided by layout
        x, y = pos_x, pos_y

        # adjust for child's top and left margins
        if hasattr(child, "style") and hasattr(child.style, "margin"):
            x += child.style.margin_left
            y += child.style.margin_top

        # --- cross-axis alignment ---
        is_text = hasattr(child, "vertical_align") and hasattr(child, "align")
        is_row = self.layout.direction == "row"

        # vertical alignment (cross axis for row layout)
        if is_row and avail_h is not None:
            child_margins_v = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                child_margins_v = child.style.margin_top + child.style.margin_bottom
            # effective available height after subtracting margins
            effective_h = avail_h - child_margins_v
            child_h = child._transformed_aabb.height

            # get alignment setting
            align_setting = self.layout.align_items
            if is_text:  # text might have its own alignment property
                align_setting = child.vertical_align  # assumes Text uses 'top', 'middle', 'bottom'

            # apply alignment (relative to effective height)
            if align_setting == "center" or align_setting == "middle":
                y = pos_y + (effective_h - child_h) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    y += child.style.margin_top  # re-add top margin
            elif align_setting == "end" or align_setting == "bottom":
                y = pos_y + effective_h - child_h
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    # position is bottom edge, adjust by bottom margin is tricky, easier to calculate from top
                    y = pos_y + avail_h - child.style.margin_bottom - child_h
            # 'start' or 'top' alignment is handled by the initial pos_y + margin_top

        # horizontal alignment (cross axis for column layout)
        elif not is_row and avail_w is not None:
            child_margins_h = 0
            if hasattr(child, "style") and hasattr(child.style, "margin"):
                child_margins_h = child.style.margin_left + child.style.margin_right
            # effective available width after subtracting margins
            effective_w = avail_w - child_margins_h
            child_w = child._transformed_aabb.width

            # get alignment setting
            align_setting = self.layout.align_items
            if is_text:  # text might have its own alignment property
                align_setting = child.align  # assumes Text uses 'left', 'center', 'right'

            # apply alignment (relative to effective width)
            if align_setting == "center":
                x = pos_x + (effective_w - child_w) / 2
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    x += child.style.margin_left  # re-add left margin
            elif align_setting == "end" or align_setting == "right":
                x = pos_x + effective_w - child_w
                if hasattr(child, "style") and hasattr(child.style, "margin"):
                    # position is right edge, adjust by right margin is tricky, easier to calculate from left
                    x = pos_x + avail_w - child.style.margin_right - child_w
            # 'start' or 'left' alignment is handled by the initial pos_x + margin_left

        # set final translation transform for the child
        child.transform.translate = (x, y)
        self._log_debug(f"Positioned child {child.id}", {"pos": (x, y)})

    # REMOVED compute_world_matrix override - use inherited version

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
            self._log_debug("Rendering container", {"matrix": matrix.tolist()})  # log as list

        regular_children = [c for c in self.children if not c.is_overlay]
        overlay_children = [c for c in self.children if c.is_overlay]

        # render regular children first
        for child in regular_children:
            # child matrix is relative to this container
            child_matrix = child.compute_local_matrix()
            # world matrix combines parent's world matrix with child's local
            world_matrix = matrix @ child_matrix
            if self.debug:
                self._log_debug(
                    f"Rendering regular child {child.id}", {"world_matrix": world_matrix.tolist()}
                )
            child.render(renderer, context, world_matrix)

        # render overlays on top
        for overlay in overlay_children:
            # overlay matrix is relative to this container
            overlay_matrix = overlay.compute_local_matrix()
            # world matrix combines parent's world matrix with overlay's local
            world_matrix = matrix @ overlay_matrix
            if self.debug:
                self._log_debug(
                    f"Rendering overlay {overlay.id}", {"world_matrix": world_matrix.tolist()}
                )
            overlay.render(renderer, context, world_matrix)

    def add_child(self, child: Component):
        """add child component and set parent link"""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            self._log_debug(f"Added child {child.id or 'Unnamed'}")

    def add_children(self, children: list[Component]):
        for child in children:
            self.add_child(child)
