"""Core data models for geometry, styling, and layout."""

from typing import Tuple, Optional, Literal, Union, TypeVar, Sequence
from pydantic import BaseModel, Field, model_validator, AliasChoices
import numpy as np
import logging

from jeanplot.debug import debug_print


logger = logging.getLogger(__name__)

LayoutDirection = Literal["row", "column"]
AlignType = Literal["start", "center", "end", "stretch"]
DistributeType = Literal["start", "center", "end", "space-between", "space-around", "space-evenly"]
LineStyleType = Literal["solid", "dashed", "dotted", "custom"]
LineWidthMode = Literal["point", "data"]

T = TypeVar("T")


class Size(BaseModel):
    """Represents a 2D size."""

    width: float = 0.0
    height: float = 0.0

    def __init__(self, width: float = 0.0, height: float = 0.0, **data):
        super().__init__(width=width, height=height, **data)

    def union(self, other: "Size") -> "Size":
        return Size(width=max(self.width, other.width), height=max(self.height, other.height))

    @classmethod
    def min(cls, size1: "Size", size2: "Size") -> "Size":
        return Size(width=min(size1.width, size2.width), height=min(size1.height, size2.height))

    @classmethod
    def max(cls, size1: "Size", size2: "Size") -> "Size":
        return Size(width=max(size1.width, size2.width), height=max(size1.height, size2.height))

    def __repr__(self) -> str:
        return f"Size(w={self.width:.1f}, h={self.height:.1f})"

    def __str__(self) -> str:
        return self.__repr__()


class Offset(BaseModel):
    """Defines an offset relative to component/parent dimensions and absolute values."""

    # % of self size (component receiving the offset)
    relative: Tuple[float, float] = (0.0, 0.0)
    # % of reference size (parent or attachment target)
    reference_relative: Tuple[float, float] = Field(
        default=(0.0, 0.0), validation_alias=AliasChoices("reference_relative", "parent_relative")
    )
    # absolute units
    absolute: Tuple[float, float] = (0.0, 0.0)

    def compute(
        self, self_dims: Size, reference_dims: Optional[Size] = None
    ) -> Tuple[float, float]:
        """calculate final offset vector based on self and reference dimensions."""
        ref_dims = reference_dims or Size()  # use zero size if no reference
        x = (
            self_dims.width * self.relative[0]
            + ref_dims.width * self.reference_relative[0]
            + self.absolute[0]
        )
        y = (
            self_dims.height * self.relative[1]
            + ref_dims.height * self.reference_relative[1]
            + self.absolute[1]
        )
        if logger.isEnabledFor(logging.DEBUG):
            debug_print(
                f"{self.__class__.__name__}.compute",
                f"self={self_dims}, ref={ref_dims}, def={self} -> ({x:.1f}, {y:.1f})",
            )
        return x, y

    def __repr__(self) -> str:
        parts = []
        if self.relative != (0.0, 0.0):
            parts.append(f"rel=({self.relative[0]:.1f},{self.relative[1]:.1f})")
        if self.reference_relative != (0.0, 0.0):
            parts.append(
                f"ref_rel=({self.reference_relative[0]:.1f},{self.reference_relative[1]:.1f})"
            )
        if self.absolute != (0.0, 0.0):
            parts.append(f"abs=({self.absolute[0]:.1f},{self.absolute[1]:.1f})")
        return f"Offset({', '.join(parts)})" if parts else "Offset()"

    def __str__(self) -> str:
        return self.__repr__()


class Transform(BaseModel):
    """Transformation matrix components."""

    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0  # degrees
    scale: Tuple[float, float] = (1.0, 1.0)
    skew_x: float = 0.0  # degrees
    skew_y: float = 0.0  # degrees
    # % of component size
    rotation_center: Tuple[float, float] = (0.5, 0.5)

    def to_matrix(self, dimensions: Size) -> np.ndarray:
        """convert to 3x3 homogeneous transform matrix."""
        # debug_print(f"{self.__class__.__name__}.to_matrix", f"dims={dimensions}, def={self}") # can be noisy

        # 1. scale
        s_mat = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
        # 2. skew
        sx_rad, sy_rad = np.radians(self.skew_x), np.radians(self.skew_y)
        skew_mat = np.array([[1, np.tan(sx_rad), 0], [np.tan(sy_rad), 1, 0], [0, 0, 1]])
        # 3. rotate (around rotation_center)
        r_mat = np.identity(3)
        if self.rotate != 0.0:
            theta = np.radians(self.rotate)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
            if dimensions.width > 0 and dimensions.height > 0:
                cx = dimensions.width * self.rotation_center[0]
                cy = dimensions.height * self.rotation_center[1]
                center_t = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                uncenter_t = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                r_mat = center_t @ rot @ uncenter_t
            else:
                r_mat = rot  # rotate around origin if no dimensions
        # 4. translate
        t_mat = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])

        # combined: T * R * Sk * Sc
        # matrix applies right-to-left: scale -> skew -> rotate -> translate
        final_matrix = t_mat @ r_mat @ skew_mat @ s_mat
        return final_matrix

    def __repr__(self) -> str:
        parts = []
        if self.translate != (0.0, 0.0):
            parts.append(f"t={self.translate}")
        if self.rotate != 0.0:
            parts.append(f"r={self.rotate}")
        if self.scale != (1.0, 1.0):
            parts.append(f"s={self.scale}")
        if self.skew_x != 0.0 or self.skew_y != 0.0:
            parts.append(f"sk=({self.skew_x},{self.skew_y})")
        if self.rotation_center != (0.5, 0.5):
            parts.append(f"rc={self.rotation_center}")
        return f"Transform({', '.join(parts)})" if parts else "Transform()"

    def __str__(self) -> str:
        return self.__repr__()


class BorderStyle(BaseModel):
    """Border styling properties."""

    border_color: Optional[str] = None
    border_width: float = 0.0
    border_width_mode: LineWidthMode = "point"  # prefer point default for consistency
    border_style: LineStyleType = "solid"
    dash_sequence: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0


class MarginPadding(BaseModel):
    """Margin and padding properties (top, right, bottom, left)."""

    margin: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

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
        """Returns the padding tuple (top, right, bottom, left)."""
        return self.padding

    def content_box(self, bounds: Size) -> Tuple[float, float]:
        """Calculates the content area size within given bounds."""
        inset = self.content_inset()
        return (
            max(0, bounds.width - inset[1] - inset[3]),
            max(0, bounds.height - inset[0] - inset[2]),
        )


class Shadow(BaseModel):
    """Shadow styling properties."""

    offset_x: float = 0.0
    offset_y: float = 0.0
    blur_radius: float = 3.0
    spread: float = 0.0
    color: str = "#00000080"
    resolution: float = 1.0  # controls number of layers in approximation


class BoxStyle(BorderStyle, MarginPadding):
    """Combined styling for borders, margins, padding, background, and shadow."""

    background_color: Optional[str] = None
    shadow: Optional[Shadow] = None


class LayoutConstraints(BaseModel):
    """Defines how children are arranged within a container."""

    direction: LayoutDirection = "row"
    align_items: AlignType = "start"
    justify_content: DistributeType = "start"
    gap: float = 0.0
    wrap: bool = False  # wrap not implemented in layout logic yet

    def __repr__(self) -> str:
        return f"Layout(dir={self.direction}, align={self.align_items}, justify={self.justify_content}, gap={self.gap:.1f})"

    def __str__(self) -> str:
        return self.__repr__()
