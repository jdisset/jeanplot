from typing import List, Optional, Union, Tuple, Dict, Any, Set, Literal
from pydantic import BaseModel, Field
import numpy as np

FlexDirectionType = Literal["row", "column"]
PositionType = Literal["relative", "absolute"]
LineStyleType = Literal["solid", "dashed", "custom"]
AlignType = Literal["left", "center", "right"]
VerticalAlignType = Literal["top", "center", "bottom"]


class Transform(BaseModel):
    """transform model handling translation, rotation and scaling"""

    translate: Tuple[float, float] = (0.0, 0.0)
    rotate: float = 0.0
    scale: Tuple[float, float] = (1.0, 1.0)

    def to_matrix(self) -> np.ndarray:
        t = np.array([[1, 0, self.translate[0]], [0, 1, self.translate[1]], [0, 0, 1]])
        theta = np.radians(self.rotate)
        r = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        s = np.array([[self.scale[0], 0, 0], [0, self.scale[1], 0], [0, 0, 1]])
        return t @ r @ s


class ContainerStyle(BaseModel):
    margin: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    padding: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: float = 0.0
    border_style: LineStyleType = "solid"
    dash_sequence: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0
    corner_radius: float = 0.0

    def get_matplotlib_linestyle(self) -> Union[str, Tuple[float, Tuple[float, ...]]]:
        if self.border_style == "solid":
            return "-"
        if self.border_style == "dashed":
            return "--"
        if self.border_style == "custom" and self.dash_sequence:
            return (self.dash_offset, self.dash_sequence)
        return "-"


class Bounds(BaseModel):
    """rectangular bounds with utility methods for containment and manipulation"""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def union(self, other: "Bounds") -> "Bounds":
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x + self.width, other.x + other.width)
        y2 = max(self.y + self.height, other.y + other.height)
        return Bounds(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def expand(self, amount: float) -> "Bounds":
        return Bounds(
            x=self.x - amount,
            y=self.y - amount,
            width=self.width + 2 * amount,
            height=self.height + 2 * amount,
        )
