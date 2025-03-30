from typing import Tuple, Optional, Literal, Union
from pydantic import BaseModel, Field
import numpy as np

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

    relative: Tuple[float, float] = (0.0, 0.0)
    absolute: Tuple[float, float] = (0.0, 0.0)

    def compute(self, dimensions: Size) -> Tuple[float, float]:
        """Compute final offset based on dimensions"""
        return (
            dimensions.width * self.relative[0] + self.absolute[0],
            dimensions.height * self.relative[1] + self.absolute[1],
        )


class Transform(BaseModel):
    """transformation matrix components"""

    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)
    rotation_center: Tuple[float, float] = (0.5, 0.5)

    def to_matrix(self, dimensions: Optional[Size] = None) -> np.ndarray:
        """convert to 3x3 homogeneous transform matrix"""
        # handle rotation center with dimensions
        if dimensions and self.rotate != 0.0:
            cx, cy = (
                dimensions.width * self.rotation_center[0],
                dimensions.height * self.rotation_center[1],
            )

            # matrices
            to_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            theta = np.radians(self.rotate)
            r = np.array(
                [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )
            from_center = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
            s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
            t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])

            # combine: t * s * from_center * r * to_center
            return t @ s @ from_center @ r @ to_center
        else:
            # simple case without dimensions or rotation
            t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])

            if self.rotate != 0.0:
                theta = np.radians(self.rotate)
                r = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1],
                    ]
                )
            else:
                r = np.eye(3)

            s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
            return t @ r @ s


class VisualStyle(BaseModel):
    margin: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: float = 0.0
    border_width_mode: LineWidthMode = "data"
    border_style: LineStyleType = "solid"
    dash_sequence: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0

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


class LayoutConstraints(BaseModel):
    direction: LayoutDirection = "row"
    align_items: AlignType = "start"
    justify_content: DistributeType = "start"
    gap: float = 0.0
    wrap: bool = False
