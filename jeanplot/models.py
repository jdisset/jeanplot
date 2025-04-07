from typing import (
    Tuple,
    Optional,
    Literal,
    Union,
    TypeVar,
)
from pydantic import BaseModel
import numpy as np


LayoutDirection = Literal["row", "column"]
AlignType = Literal["start", "center", "end", "stretch"]
DistributeType = Literal["start", "center", "end", "space-between", "space-around", "space-evenly"]
LineStyleType = Literal["solid", "dashed", "dotted", "custom"]
LineWidthMode = Literal["point", "data"]

T = TypeVar("T")


class Size(BaseModel):
    width: float = 0.0
    height: float = 0.0

    def __init__(self, width=0.0, height=0.0, **data):
        super().__init__(width=width, height=height, **data)

    def union(self, other: "Size") -> "Size":
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
    parent_relative: Tuple[float, float] = (0.0, 0.0)  # relative to parent/target dimensions
    absolute: Tuple[float, float] = (0.0, 0.0)  # absolute offset

    def compute(
        self, dimensions: Size, target_dimensions: Optional[Size] = None
    ) -> Tuple[float, float]:
        """Compute final offset based on self and target dimensions"""
        x = dimensions.width * self.relative[0]
        y = dimensions.height * self.relative[1]
        if target_dimensions and (self.parent_relative[0] != 0 or self.parent_relative[1] != 0):
            x += target_dimensions.width * self.parent_relative[0]
            y += target_dimensions.height * self.parent_relative[1]
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
        """convert to 3x3 homogeneous transform matrix. applies scale, skew, rotate (around center), then translate."""
        s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
        sx_rad, sy_rad = np.radians(self.skew_x), np.radians(self.skew_y)
        skew_m = np.array([[1, np.tan(sx_rad), 0], [np.tan(sy_rad), 1, 0], [0, 0, 1]])
        theta = np.radians(self.rotate)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])

        matrix = np.identity(3)
        matrix = s @ matrix
        matrix = skew_m @ matrix

        if self.rotate != 0.0 and dimensions is not None:
            cx = dimensions.width * self.rotation_center[0]
            cy = dimensions.height * self.rotation_center[1]
            to_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            from_center = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
            matrix = from_center @ r @ to_center @ matrix
        elif self.rotate != 0.0:  # rotation without centering
            matrix = r @ matrix

        t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])
        matrix = t @ matrix
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
    margin_top = property(lambda self: self.margin[0])
    margin_right = property(lambda self: self.margin[1])
    margin_bottom = property(lambda self: self.margin[2])
    margin_left = property(lambda self: self.margin[3])
    padding_top = property(lambda self: self.padding[0])
    padding_right = property(lambda self: self.padding[1])
    padding_bottom = property(lambda self: self.padding[2])
    padding_left = property(lambda self: self.padding[3])

    def content_inset(self) -> Tuple[float, float, float, float]:
        return self.padding

    def content_box(self, bounds: Size) -> Tuple[float, float]:
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
