from typing import Tuple, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator
import numpy as np
import math

# type definitions
LayoutDirection = Literal["row", "column"]
AlignType = Literal["start", "center", "end", "stretch"]
DistributeType = Literal["start", "center", "end", "space-between", "space-around", "space-evenly"]
LineStyleType = Literal["solid", "dashed", "dotted", "custom"]
LineWidthMode = Literal["point", "data"]


class Size(BaseModel):
    width: float = 0.0
    height: float = 0.0

    def __init__(self, width=0.0, height=0.0, **data):
        super().__init__(width=width, height=height, **data)

    def union(self, other: "Size") -> "Size":
        """union of two bounds"""
        return Size(width=max(self.width, other.width), height=max(self.height, other.height))

    @classmethod
    def min(cls, size1: "Size", size2: "Size") -> "Size":
        return Size(width=min(size1.width, size2.width), height=min(size1.height, size2.height))

    @classmethod
    def max(cls, size1: "Size", size2: "Size") -> "Size":
        return Size(width=max(size1.width, size2.width), height=max(size1.height, size2.height))

    def __gt__(self, other: Union["Size", float]) -> bool:
        if isinstance(other, Size):
            return self.width > other.width and self.height > other.height
        return self.width > other and self.height > other

    def __lt__(self, other: Union["Size", float]) -> bool:
        if isinstance(other, Size):
            return self.width < other.width and self.height < other.height
        return self.width < other and self.height < other


class Offset(BaseModel):
    """Unified offset with relative and absolute components"""

    relative: Tuple[float, float] = (0.0, 0.0)  # relative to self dimensions
    parent_relative: Tuple[float, float] = (0.0, 0.0)  # relative to parent dimensions
    absolute: Tuple[float, float] = (0.0, 0.0)  # absolute offset

    def compute(
        self, dimensions: Size, parent_dimensions: Optional[Size] = None
    ) -> Tuple[float, float]:
        """Compute final offset based on dimensions and parent dimensions"""
        x = dimensions.width * self.relative[0]
        y = dimensions.height * self.relative[1]

        if parent_dimensions and (self.parent_relative[0] != 0 or self.parent_relative[1] != 0):
            x += parent_dimensions.width * self.parent_relative[0]
            y += parent_dimensions.height * self.parent_relative[1]

        x += self.absolute[0]
        y += self.absolute[1]

        return (x, y)


class Transform(BaseModel):
    """transformation matrix components"""

    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0  # degrees
    scale: Tuple[float, float] = (1.0, 1.0)
    skew_x: float = 0.0
    skew_y: float = 0.0
    rotation_center: Tuple[float, float] = (0.5, 0.5)

    def to_matrix(self, dimensions: Optional[Size] = None) -> np.ndarray:
        """convert to 3x3 homogeneous transform matrix"""
        # translation matrix
        t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])

        # scaling matrix
        s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])

        # skew matrix
        # convert degrees to radians for tan
        sx_rad = np.radians(self.skew_x)
        sy_rad = np.radians(self.skew_y)
        tan_x = np.tan(sx_rad)
        tan_y = np.tan(sy_rad)
        # combined skew matrix (applies x-skew based on y, y-skew based on x)
        skew_m = np.array([[1, tan_x, 0], [tan_y, 1, 0], [0, 0, 1]])

        # rotation matrix and centering logic
        theta = np.radians(self.rotate)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        r = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])

        # if rotation is non-zero AND dimensions are provided for centering
        if (
            self.rotate != 0.0
            and dimensions
            and (self.rotation_center[0] != 0.0 or self.rotation_center[1] != 0.0)
        ):
            # calculate center point in local coordinates
            cx = dimensions.width * self.rotation_center[0]
            cy = dimensions.height * self.rotation_center[1]

            # matrices for moving rotation center to origin and back
            to_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            from_center = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

            # combined transformation: Translate * Scale * Skew * MoveToOrigin * Rotate * MoveFromOrigin
            # note: applied right-to-left -> matrix multiplication left-to-right
            # applying skew *before* the rotation centering logic
            # matrix = t @ s @ skew_m @ from_center @ r @ to_center
            # let's try applying skew *after* scale but before rotation
            matrix = t @ from_center @ r @ to_center @ s @ skew_m

        else:
            # simpler case: rotation around origin (0,0) or no rotation
            # apply T * R * S * Skew (relative to origin)
            # if rotation is 0, R is identity
            matrix = t @ r @ s @ skew_m

        return matrix


class BorderStyle(BaseModel):
    border_color: Optional[str] = None
    border_width: float = 0.0
    border_width_mode: LineWidthMode = "data"
    border_style: LineStyleType = "solid"
    dash_sequence: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0


class MarginPadding(BaseModel):
    margin: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # margin accessors
    margin_top = property(lambda self: self.margin[0])
    margin_right = property(lambda self: self.margin[1])
    margin_bottom = property(lambda self: self.margin[2])
    margin_left = property(lambda self: self.margin[3])

    # padding accessors
    padding_top = property(lambda self: self.padding[0])
    padding_right = property(lambda self: self.padding[1])
    padding_bottom = property(lambda self: self.padding[2])
    padding_left = property(lambda self: self.padding[3])

    def content_inset(self) -> Tuple[float, float, float, float]:
        return self.padding

    def content_box(self, bounds: Size) -> Tuple[float, float]:
        """return content box size after insets"""
        inset = self.content_inset()
        return (
            max(0, bounds.width - inset[1] - inset[3]),
            max(0, bounds.height - inset[0] - inset[2]),
        )


class Shadow(BaseModel):
    offset_x: float = 0.0
    offset_y: float = 0.0
    blur_radius: float = 3.0
    spread: float = 0.0
    color: str = "#00000080"
    resolution: float = 1.0


class BoxStyle(BorderStyle, MarginPadding):
    background_color: Optional[str] = None
    shadow: Optional[Shadow] = None


class LayoutConstraints(BaseModel):
    direction: LayoutDirection = "row"
    align_items: AlignType = "start"
    justify_content: DistributeType = "start"
    gap: float = 0.0
    wrap: bool = False


class AnchorPoint(BaseModel):
    """connection point with direction and min segment"""

    offset: Offset
    direction: tuple[float, float] = (0, 1)  # outward-pointing vector
    min_segment: float = 10.0  # min segment length or control point distance

    @model_validator(mode="after")
    def normalize_direction(self):
        """normalize direction vector"""
        dx, dy = self.direction
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            self.direction = (dx / length, dy / length)
        return self

    def get_position(self, component):
        """get local position"""
        dims = getattr(component, "_dimensions", Size(width=1, height=1))
        return self.offset.compute(dims)

    def get_world_position(self, component):
        """get world position"""
        ox, oy = self.get_position(component)
        local_point = np.array([ox, oy, 1])
        world_matrix = component.compute_world_matrix()
        world_point = world_matrix @ local_point
        return (world_point[0], world_point[1])
