from typing import List, Optional, Dict, Any, Union, Tuple, cast, TYPE_CHECKING, Literal
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import numpy as np
from pathlib import Path
from collections import defaultdict
from .models import Transform, Size, VisualStyle, LayoutConstraints, Offset
from functools import partial


class Component(BaseModel):
    """base component class - anything that can be rendered"""

    id: Optional[str] = None
    transform: Transform = Field(default_factory=Transform)
    offset: Offset = Field(default_factory=Offset)

    min_dimensions: Size = Field(default_factory=Size)
    max_dimensions: Size = Field(
        default_factory=partial(Size, width=float("inf"), height=float("inf"))
    )

    style: VisualStyle = Field(default_factory=VisualStyle)

    renderer_options: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))
    debug: bool = False

    _dimensions: Size = PrivateAttr(default_factory=Size)
    _transformed_aabb: Size = PrivateAttr(default_factory=Size)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.min_dimensions.width > self.max_dimensions.width:
            raise ValueError("min_dimensions.width cannot be greater than max_dimensions.width")
        if self.min_dimensions.height > self.max_dimensions.height:
            raise ValueError("min_dimensions.height cannot be greater than max_dimensions.height")
        return self

    def compute_layout_matrix(self) -> np.ndarray:
        """compute transform matrix for layout only (without offset)"""
        return self.transform.to_matrix(self._dimensions)

    def compute_local_matrix(self) -> np.ndarray:
        """convert local transform to matrix with offset applied after transform"""
        transform_matrix = self.transform.to_matrix(self._dimensions)

        # get offset in data units
        offset_x, offset_y = self.offset.compute(self._dimensions)
        offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

        # combine: offset * transform. Offset is applied after transform
        return offset_matrix @ transform_matrix

    def compute_world_matrix(self, parent_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """compute world transform by combining with parent"""
        local = self.compute_local_matrix()
        if parent_matrix is not None:
            return parent_matrix @ local
        return local

    def compute_transformed_aabb(self) -> Size:
        """compute axis-aligned bounding box after applying local transform"""
        # get the local transform matrix
        matrix = self.compute_local_matrix()

        # define corners of the component in local space
        corners = np.array(
            [
                [0, 0, 1],  # bottom left
                [self._dimensions.width, 0, 1],  # bottom right
                [0, self._dimensions.height, 1],  # top left
                [self._dimensions.width, self._dimensions.height, 1],  # top right
            ]
        )

        transformed_corners = np.dot(matrix, corners.T).T

        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])

        # return aabb dimensions
        return Size(width=max_x - min_x, height=max_y - min_y)

    def measure(self, renderer=None) -> Size:
        """measure intrinsic size if needed"""
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def apply_layout(self):
        """apply layout to position children"""
        pass

    def measure_and_layout(self, renderer=None) -> Size:
        """unified method for measurement and layout"""
        # base component just measures itself
        self.measure(renderer)
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render this component using provided renderer"""
        raise NotImplementedError("Subclasses must implement render")

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        """add a renderer-specific option"""
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> Dict[str, Any]:
        """get all options for a specific renderer"""
        return self.renderer_options.get(renderer_name, {})


class Container(Component):
    """container that lays out and renders child components"""

    children: List[Component] = []
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

            # render debug visuals if needed
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

        # render debug visuals if needed
        if self.debug:
            renderer.render_debug(context, self, matrix)

        # render all children with combined transform
        for child in self.children:
            child_matrix = child.compute_local_matrix()
            world_matrix = matrix @ child_matrix
            child.render(renderer, context, world_matrix)


class SVGElement(Component):
    """svg element loaded from a file"""

    file_path: Union[str, Path]
    main_color: str = "black"
    secondary_color: str = "gray"

    svg_data: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self.load_svg()

    def load_svg(self):
        """load svg file and extract paths and viewBox"""
        try:
            from lxml import etree

            path = Path(self.file_path)
            tree = etree.parse(str(path))
            root = tree.getroot()

            # extract size and viewBox
            try:
                width = float(root.attrib.get("width", "100").rstrip("px"))
                height = float(root.attrib.get("height", "100").rstrip("px"))
                self._dimensions.width = width
                self._dimensions.height = height
            except (ValueError, TypeError):
                # fallback sizes if parsing fails
                self._dimensions.width = 100
                self._dimensions.height = 100

            # extract viewBox if present
            self.svg_data["viewBox"] = None
            if "viewBox" in root.attrib:
                try:
                    viewBox = [float(x) for x in root.attrib["viewBox"].split()]
                    if len(viewBox) == 4:
                        self.svg_data["viewBox"] = tuple(viewBox)
                except (ValueError, TypeError):
                    pass

            # extract path data
            self.svg_data["paths"] = []
            for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
                path_data = {
                    "d": path.attrib.get("d", ""),
                    "fill": path.attrib.get("fill", "none"),
                    "stroke": path.attrib.get("stroke", "none"),
                    "stroke_width": float(path.attrib.get("stroke-width", 1.0)),
                }

                # track color attributes for later customization
                if path_data["fill"] == "#0000FF":
                    path_data["is_main_color"] = True
                elif path_data["fill"] == "#00FF00":
                    path_data["is_secondary_color"] = True

                self.svg_data["paths"].append(path_data)

            # calculate transformed aabb
            self._transformed_aabb = self.compute_transformed_aabb()

        except Exception as e:
            raise RuntimeError(f"Failed to load SVG file at {self.file_path}: {e}")

    def measure(self, renderer=None) -> Size:
        """return the natural size of the svg"""
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render svg using the provided renderer"""
        renderer.render_svg(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)


class Text(Component):
    """text component rendered via svg path conversion"""

    text: str
    font_name: Optional[str] = None  # use default font when None
    font_size: float = 12.0  # in data units
    font_weight: Literal["normal", "bold"] = "normal"
    font_style: Literal["normal", "italic"] = "normal"
    color: str = "black"
    align: Literal["left", "center", "right"] = "left"
    vertical_align: Literal["top", "middle", "bottom"] = "top"

    # cache for text path data
    _text_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def measure(self, renderer=None) -> Size:
        """measure text dimensions using renderer"""
        if not self.text:
            self._dimensions = Size(width=0, height=0)
            self._transformed_aabb = Size(width=0, height=0)
            return self._dimensions

        if renderer and hasattr(renderer, "measure_text"):
            # use renderer's measurement capability for exact dimensions
            measured_size = renderer.measure_text(self)
            self._dimensions = Size(
                width=max(self.min_dimensions.width, measured_size.width),
                height=max(self.min_dimensions.height, measured_size.height),
            )
        else:
            # if no renderer is available, use minimum dimensions
            self._dimensions = Size(
                width=self.min_dimensions.width, height=self.min_dimensions.height
            )

        # apply max constraints
        self._dimensions = Size(
            width=min(self._dimensions.width, self.max_dimensions.width),
            height=min(self._dimensions.height, self.max_dimensions.height),
        )

        self._transformed_aabb = self.compute_transformed_aabb()

        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render text component using renderer"""
        renderer.render_text(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
