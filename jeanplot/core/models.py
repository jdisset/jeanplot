# File: jeanplot/models.py
# -*- coding: utf-8 -*-
"""Core data models for geometry, styling, and layout."""

from typing import Literal, TypeVar, Annotated
from pydantic import BaseModel, Field, AliasChoices, BeforeValidator
import numpy as np
import logging

# use absolute imports

logger = logging.getLogger(__name__)


def normalize_color(color: str | tuple | None) -> str | None:
    # handles tuple colors too now
    if not color or (isinstance(color, str) and color.lower() == "none"):
        return None
    try:
        from matplotlib.colors import to_hex  # local import ok here

        return to_hex(color, keep_alpha=True)
    except (ValueError, TypeError, AttributeError):  # added attributeerror
        logger.warning(f"could not normalize color '{color}', returning None.")
        return None


# types
LayoutDirection = Literal["row", "column"]
AlignType = Literal["start", "center", "end", "stretch"]
DistributeType = Literal["start", "center", "end", "space-between", "space-around", "space-evenly"]
LineStyleType = Literal["solid", "dashed", "dotted", "custom"]
LineWidthMode = Literal["point", "data"]
NormalizedColor = Annotated[str | None, BeforeValidator(normalize_color)]

T = TypeVar("T")


class Size(BaseModel):
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
    relative: tuple[float, float] = (0.0, 0.0)
    reference_relative: tuple[float, float] = Field(
        default=(0.0, 0.0), validation_alias=AliasChoices("reference_relative", "parent_relative")
    )
    absolute: tuple[float, float] = (0.0, 0.0)

    def compute(self, self_dims: Size, reference_dims: Size | None = None) -> tuple[float, float]:
        ref_dims = reference_dims or Size()
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
    translate: tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0  # degrees
    scale: tuple[float, float] = (1.0, 1.0)
    skew_x: float = 0.0  # degrees
    skew_y: float = 0.0  # degrees
    rotation_center: tuple[float, float] = (0.5, 0.5)

    def to_matrix(self, dimensions: Size) -> np.ndarray:
        s_mat = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
        sx_rad, sy_rad = np.radians(self.skew_x), np.radians(self.skew_y)
        skew_mat = np.array([[1, np.tan(sx_rad), 0], [np.tan(sy_rad), 1, 0], [0, 0, 1]])
        r_mat = np.identity(3)
        if abs(self.rotate) > 1e-6:
            theta = np.radians(self.rotate)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
            if dimensions.width > 1e-6 and dimensions.height > 1e-6:
                cx = dimensions.width * self.rotation_center[0]
                cy = dimensions.height * self.rotation_center[1]
                center_t = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                uncenter_t = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                r_mat = center_t @ rot @ uncenter_t
            else:
                r_mat = rot
        t_mat = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])
        return t_mat @ r_mat @ skew_mat @ s_mat

    def __repr__(self) -> str:
        parts = []
        if self.translate != (0.0, 0.0):
            parts.append(f"t={self.translate}")
        if abs(self.rotate) > 1e-6:
            parts.append(f"r={self.rotate:.1f}")
        if self.scale != (1.0, 1.0):
            parts.append(f"s={self.scale}")
        if abs(self.skew_x) > 1e-6 or abs(self.skew_y) > 1e-6:
            parts.append(f"sk=({self.skew_x:.1f},{self.skew_y:.1f})")
        if self.rotation_center != (0.5, 0.5):
            parts.append(f"rc={self.rotation_center}")
        return f"Transform({', '.join(parts)})" if parts else "Transform()"

    def __str__(self) -> str:
        return self.__repr__()


class BorderStyle(BaseModel):
    border_color: NormalizedColor = None
    border_width: float = 0.0
    border_width_mode: LineWidthMode = "data"
    border_style: LineStyleType = "solid"
    dash_sequence: tuple[float, ...] | None = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0


class MarginPadding(BaseModel):
    margin: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

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

    def content_inset(self) -> tuple[float, float, float, float]:
        return self.padding

    def content_box(self, bounds: Size) -> tuple[float, float]:
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
    color: NormalizedColor = "#00000080"
    resolution: float = 1.0


class BoxStyle(BorderStyle, MarginPadding):
    background_color: NormalizedColor = None
    shadow: Shadow | None = None


class LayoutConstraints(BaseModel):
    direction: LayoutDirection = "row"
    align_items: AlignType = "start"
    justify_content: DistributeType = "start"
    gap: float = 0.0
    wrap: bool = False

    def __repr__(self) -> str:
        return f"Layout(dir={self.direction}, align={self.align_items}, justify={self.justify_content}, gap={self.gap:.1f})"

    def __str__(self) -> str:
        return self.__repr__()


class TextMetrics(BaseModel):
    """stores measured text metrics at a reference size."""

    ref_font_size: float = 10.0
    width_points: float = 0.0
    height_points: float = 0.0
