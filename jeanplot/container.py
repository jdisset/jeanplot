from pydantic import Field
from .component import Component
import numpy as np
from .models import Size, LayoutConstraints


class Container(Component):
    """container that lays out and renders child components"""

    children: list[Component] = []
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)

    def measure(self, renderer=None) -> Size:
        """measure this container based on children's bounds"""

        if not self.children:
            self._dimensions = self.min_dimensions
            self._transformed_aabb = self.compute_transformed_aabb()
            return self._dimensions

        if self.layout.direction == "row":
            total_width = sum(child._transformed_aabb.width for child in self.children)
            total_width += self.layout.gap * (len(self.children) - 1)
            max_height = max((child._transformed_aabb.height for child in self.children), default=0)

            width = max(self.min_dimensions.width, total_width)
            height = max(self.min_dimensions.height, max_height)

        else:  # column
            total_height = sum(child._transformed_aabb.height for child in self.children)
            total_height += self.layout.gap * (len(self.children) - 1)
            max_width = max((child._transformed_aabb.width for child in self.children), default=0)

            width = max(self.min_dimensions.width, max_width)
            height = max(self.min_dimensions.height, total_height)

        # add padding and margin
        insets = self.style.content_inset()
        width += insets[1] + insets[3]  # right + left
        height += insets[0] + insets[2]  # top + bottom

        # apply min/max constraints
        self._dimensions = Size(
            width=min(max(self.min_dimensions.width, width), self.max_dimensions.width),
            height=min(max(self.min_dimensions.height, height), self.max_dimensions.height),
        )

        self._transformed_aabb = self.compute_transformed_aabb()

        return self._dimensions

    def measure_and_layout(self, renderer=None) -> Size:
        """unified method for measurement and layout"""
        # first process all children (bottom-up)
        for child in self.children:
            child.measure_and_layout(renderer)

        self.measure(renderer)

        # apply layout to position children
        self.apply_layout()

        return self._dimensions

    def apply_layout(self):
        """apply layout to position children based on container dimensions"""
        if not self.children:
            return

        content_width, content_height = self.style.content_box(self._dimensions)

        # apply insets to get content origin
        insets = self.style.content_inset()
        content_x = insets[3]  # left inset
        content_y = insets[0]  # top inset

        # position children based on layout direction
        if self.layout.direction == "row":
            self._row_layout(content_x, content_y, content_width, content_height)
        else:  # column
            self._column_layout(content_x, content_y, content_width, content_height)

        # after layout, check if any children need to be resized (stretch alignment)
        # and if so, remeasure this container
        stretched_containers = self._apply_stretch_alignment()
        if stretched_containers:
            self.measure()

            # reapply layout to any stretched containers with children
            for container in stretched_containers:
                if isinstance(container, Container) and container.children:
                    container.apply_layout()

    def _apply_stretch_alignment(self):
        """apply stretch alignment and return list of stretched components"""
        if not self.children:
            return []

        stretched_containers = []
        content_width, content_height = self.style.content_box(self._dimensions)

        # for row layout, stretch height
        if self.layout.direction == "row" and self.layout.align_items == "stretch":
            for child in self.children:
                if (
                    child._dimensions.height < content_height
                    and child._dimensions.height < child.max_dimensions.height
                ):
                    old_height = child._dimensions.height
                    child._dimensions.height = min(content_height, child.max_dimensions.height)
                    if old_height != child._dimensions.height:
                        # recalculate transformed aabb
                        child._transformed_aabb = child.compute_transformed_aabb()
                        stretched_containers.append(child)

        # for column layout, stretch width
        elif self.layout.direction == "column" and self.layout.align_items == "stretch":
            for child in self.children:
                if (
                    child._dimensions.width < content_width
                    and child._dimensions.width < child.max_dimensions.width
                ):
                    old_width = child._dimensions.width
                    child._dimensions.width = min(content_width, child.max_dimensions.width)
                    if old_width != child._dimensions.width:
                        # recalculate transformed aabb
                        child._transformed_aabb = child.compute_transformed_aabb()
                        stretched_containers.append(child)

        return stretched_containers

    def _position_child(self, child, pos_x, pos_y, available_height=None, available_width=None):
        """helper method to position a child based on alignment"""
        # start with default position
        adjusted_x = pos_x
        adjusted_y = pos_y

        is_text_component = hasattr(child, "vertical_align") and hasattr(child, "align")

        # handle vertical alignment for row layout
        if self.layout.direction == "row" and available_height is not None:
            if is_text_component:
                # text component - use its own vertical_align property
                if child.vertical_align == "top":
                    adjusted_y = pos_y + available_height - child._transformed_aabb.height
                elif child.vertical_align == "middle":
                    adjusted_y = pos_y + (available_height - child._transformed_aabb.height) / 2
                elif child.vertical_align == "bottom":
                    adjusted_y = pos_y
            else:
                # standard component - use container's align_items
                if self.layout.align_items == "center":
                    adjusted_y = pos_y + (available_height - child._transformed_aabb.height) / 2
                elif self.layout.align_items == "end":
                    adjusted_y = pos_y + available_height - child._transformed_aabb.height
                # "start" and "stretch" use the default pos_y

        # handle horizontal alignment for column layout
        if self.layout.direction == "column" and available_width is not None:
            if is_text_component:
                # text component - use its own align property
                if child.align == "left":
                    adjusted_x = pos_x
                elif child.align == "center":
                    adjusted_x = pos_x + (available_width - child._transformed_aabb.width) / 2
                elif child.align == "right":
                    adjusted_x = pos_x + available_width - child._transformed_aabb.width
            else:
                # standard component - use container's align_items
                if self.layout.align_items == "center":
                    adjusted_x = pos_x + (available_width - child._transformed_aabb.width) / 2
                elif self.layout.align_items == "end":
                    adjusted_x = pos_x + available_width - child._transformed_aabb.width
                # "start" and "stretch" use the default pos_x

        # set the position via the transform's translate property
        # important: this is the layout position, offset will be applied separately
        child.transform.translate = (adjusted_x, adjusted_y)

    def _row_layout(self, content_x, content_y, content_width, content_height):
        """layout children in a row"""
        # filter zero-width children
        valid_children = [child for child in self.children if child._transformed_aabb.width > 0]

        if not valid_children:
            return

        # calculate total width and space
        total_width = sum(child._transformed_aabb.width for child in valid_children)
        total_gaps = self.layout.gap * (len(valid_children) - 1)
        total_required_width = total_width + total_gaps

        extra_space = max(0, content_width - total_required_width)

        # starting position based on justification
        start_x = content_x
        if self.layout.justify_content == "center":
            start_x += extra_space / 2
        elif self.layout.justify_content == "end":
            start_x += extra_space

        # spacing for distributed layouts
        item_spacing = 0
        if len(valid_children) > 1:
            if self.layout.justify_content == "space-between":
                item_spacing = extra_space / (len(valid_children) - 1)
            elif self.layout.justify_content == "space-around":
                item_spacing = extra_space / len(valid_children)
                start_x += item_spacing / 2
            elif self.layout.justify_content == "space-evenly":
                item_spacing = extra_space / (len(valid_children) + 1)
                start_x += item_spacing

        # position each child
        current_x = start_x
        for child in valid_children:
            self._position_child(child, current_x, content_y, available_height=content_height)
            current_x += child._transformed_aabb.width + self.layout.gap + item_spacing

        # handle zero-width children
        for child in self.children:
            if child._transformed_aabb.width <= 0:
                self._position_child(child, start_x, content_y, available_height=content_height)

    def _column_layout(self, content_x, content_y, content_width, content_height):
        """layout children in a column"""
        # filter zero-height children
        valid_children = [child for child in self.children if child._transformed_aabb.height > 0]

        if not valid_children:
            return

        # calculate total height and space
        total_height = sum(child._transformed_aabb.height for child in valid_children)
        total_gaps = self.layout.gap * (len(valid_children) - 1)
        total_required_height = total_height + total_gaps

        extra_space = max(0, content_height - total_required_height)

        # starting position based on justification
        start_y = content_y
        if self.layout.justify_content == "center":
            start_y += extra_space / 2
        elif self.layout.justify_content == "end":
            start_y += extra_space

        # spacing for distributed layouts
        item_spacing = 0
        if len(valid_children) > 1:
            if self.layout.justify_content == "space-between":
                item_spacing = extra_space / (len(valid_children) - 1)
            elif self.layout.justify_content == "space-around":
                item_spacing = extra_space / len(valid_children)
                start_y += item_spacing / 2
            elif self.layout.justify_content == "space-evenly":
                item_spacing = extra_space / (len(valid_children) + 1)
                start_y += item_spacing

        # position each child
        current_y = start_y
        for child in valid_children:
            self._position_child(child, content_x, current_y, available_width=content_width)
            current_y += child._transformed_aabb.height + self.layout.gap + item_spacing

        # handle zero-height children
        for child in self.children:
            if child._transformed_aabb.height <= 0:
                self._position_child(child, content_x, start_y, available_width=content_width)

    def render(self, renderer, context, matrix: np.ndarray):
        """render this container and all its children"""
        # ensure measurement and layout are current
        if not self.children:
            # simple case - just draw this container
            if self.style.background_color or self.style.border_color:
                renderer.render_rectangle(
                    context,
                    self._dimensions,
                    {
                        "background_color": self.style.background_color,
                        "border_color": self.style.border_color,
                        "width": self.style.border_width,
                        "border_width_mode": self.style.border_width_mode,
                        "corner_radius": self.style.corner_radius,
                        "border_style": self.style.border_style,
                        "dash_sequence": self.style.dash_sequence,
                        "dash_offset": self.style.dash_offset,
                    },
                    matrix,
                    component=self,
                )

            if self.debug:
                renderer.render_debug(context, self, matrix)

            return

        # render container background and border if needed
        if self.style.background_color or self.style.border_color:
            renderer.render_rectangle(
                context,
                self._dimensions,
                {
                    "background_color": self.style.background_color,
                    "border_color": self.style.border_color,
                    "width": self.style.border_width,
                    "border_width_mode": self.style.border_width_mode,
                    "corner_radius": self.style.corner_radius,
                    "border_style": self.style.border_style,
                    "dash_sequence": self.style.dash_sequence,
                    "dash_offset": self.style.dash_offset,
                },
                matrix,
                component=self,
            )

        if self.debug:
            renderer.render_debug(context, self, matrix)

        # render all children with combined transform
        for child in self.children:
            child_matrix = child.compute_local_matrix()
            world_matrix = matrix @ child_matrix
            child.render(renderer, context, world_matrix)

    def add_child(self, child: Component):
        """add a child component to this container"""
        self.children.append(child)
        if hasattr(child, "parent"):
            child.parent = self
