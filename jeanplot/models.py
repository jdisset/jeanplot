# jeanplot/models.py
from typing import Tuple, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field
import numpy as np

# --- Other models remain the same ---
LayoutDirection = Literal["row", "column"]
PositionType = Literal["relative", "absolute"]
LineStyleType = Literal["solid", "dashed", "dotted", "custom"]
AlignType = Literal["start", "center", "end", "stretch"]
DistributeType = Literal["start", "center", "end", "space-between", "space-around", "space-evenly"]
LineWidthMode = Literal["point", "data"]  # point = constant visual size, data = scales with data


class Size(BaseModel):
    width: float = 0.0
    height: float = 0.0

    def __init__(self, width=0.0, height=0.0, **data):
        if data and not all(k in ["width", "height"] for k in data.keys()):
            super().__init__(width=width, height=height, **data)
        else:
            super().__init__(width=width, height=height)

    def union(self, other: "Size") -> "Size":
        """union of two bounds"""
        return Size(width=max(self.width, other.width), height=max(self.height, other.height))

    @classmethod
    def min(cls, size1: "Size", size2: "Size") -> "Size":
        """minimum of two sizes"""
        return Size(width=min(size1.width, size2.width), height=min(size1.height, size2.height))

    @classmethod
    def max(cls, size1: "Size", size2: "Size") -> "Size":
        """maximum of two sizes"""
        return Size(width=max(size1.width, size2.width), height=max(size1.height, size2.height))

    # add a gt lt eq method for size comparison
    def __gt__(self, other: Union["Size", float]) -> bool:
        if isinstance(other, Size):
            return self.width > other.width and self.height > other.height
        elif isinstance(other, float):
            return self.width > other and self.height > other
        return False

    def __lt__(self, other: Union["Size", float]) -> bool:
        if isinstance(other, Size):
            return self.width < other.width and self.height < other.height
        elif isinstance(other, float):
            return self.width < other and self.height < other
        return False


class Offset(BaseModel):
    """
    Unified offset specifying a shift applied *after* layout positioning.
    Both relative and absolute components are added together.
    """

    relative: Tuple[float, float] = (0.0, 0.0)  # proportion of component dimensions (width, height)
    absolute: Tuple[float, float] = (0.0, 0.0)  # absolute units

    def compute(self, dimensions: Size) -> Tuple[float, float]:
        """
        Compute the combined offset shift based on component dimensions.
        Positive values shift right/down.
        """
        return (
            dimensions.width * self.relative[0] + self.absolute[0],
            dimensions.height * self.relative[1] + self.absolute[1],
        )


class Transform(BaseModel):
    """transformation matrix components (no position in bounds!)"""

    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0  # in degrees
    scale: Tuple[float, float] = (1.0, 1.0)
    rotation_center: Tuple[float, float] = (
        0.5,
        0.5,
    )  # relative coordinates (0-1), default to center

    def to_matrix(self, dimensions: Optional[Size] = None) -> np.ndarray:
        """convert to 3x3 homogeneous transform matrix"""
        # handle rotation center if dimensions are provided and rotation is non-zero
        if dimensions and self.rotate != 0.0:
            # calculate rotation center in data units
            rot_center_x = dimensions.width * self.rotation_center[0]
            rot_center_y = dimensions.height * self.rotation_center[1]

            # translate to rotation center
            t_rot_center = np.array([[1, 0, -rot_center_x], [0, 1, -rot_center_y], [0, 0, 1]])

            # apply rotation
            theta = np.radians(self.rotate)
            r = np.array(
                [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )

            # translate back from rotation center
            t_rot_back = np.array([[1, 0, rot_center_x], [0, 1, rot_center_y], [0, 0, 1]])

            s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
            t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])

            # combine: translation * scale * rotate_back * rotation * rotate_center
            # note: transforms apply right to left: first center, then rotate, then uncenter, then scale, then translate
            return t @ s @ t_rot_back @ r @ t_rot_center
        else:
            # simple case: no dimensions available or no rotation
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

            # note: transforms apply right to left: first scale, then rotate, then translate
            return t @ r @ s


class VisualStyle(BaseModel):
    margin: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # top, right, bottom, left
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # top, right, bottom, left
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: float = 0.0
    border_width_mode: LineWidthMode = "data"  # default to data units (not matplotlib default)
    border_style: LineStyleType = "solid"
    dash_sequence: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0

    @property
    def margin_top(self) -> float:
        return self.margin[0]

    @property
    def margin_right(self) -> float:
        return self.margin[1]

    @property
    def margin_bottom(self) -> float:
        return self.margin[2]

    @property
    def margin_left(self) -> float:
        return self.margin[3]

    @property
    def padding_top(self) -> float:
        return self.padding[0]

    @property
    def padding_right(self) -> float:
        return self.padding[1]

    @property
    def padding_bottom(self) -> float:
        return self.padding[2]

    @property
    def padding_left(self) -> float:
        return self.padding[3]

    def content_inset(self) -> Tuple[float, float, float, float]:
        return self.padding

    def content_box(self, bounds: Size) -> Tuple[float, float]:
        """return the size of the content box after insets"""
        inset = self.content_inset()
        return (
            max(0, bounds.width - inset[1] - inset[3]),  # width minus right and left insets
            max(0, bounds.height - inset[0] - inset[2]),  # height minus top and bottom insets
        )


class LayoutConstraints(BaseModel):
    direction: LayoutDirection = "row"
    align_items: AlignType = "start"
    justify_content: DistributeType = "start"
    gap: float = 0.0
    wrap: bool = False
