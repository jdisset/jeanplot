from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import numpy as np
from collections import defaultdict
from functools import partial

from .models import Transform, Size, VisualStyle, Offset


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

    is_overlay: bool = False
    parent: Optional["Component"] = None

    _dimensions: Size = PrivateAttr(default_factory=Size)
    _transformed_aabb: Size = PrivateAttr(default_factory=Size)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.min_dimensions.width > self.max_dimensions.width:
            raise ValueError("min_dimensions.width cannot exceed max_dimensions.width")
        if self.min_dimensions.height > self.max_dimensions.height:
            raise ValueError("min_dimensions.height cannot exceed max_dimensions.height")
        return self

    def compute_layout_matrix(self) -> np.ndarray:
        """compute transform matrix for layout only (no offset)"""
        return self.transform.to_matrix(self._dimensions)

    def compute_local_matrix(self) -> np.ndarray:
        """convert local transform to matrix with offset applied after transform"""
        transform_matrix = self.transform.to_matrix(self._dimensions)
        ox, oy = self.offset.compute(self._dimensions)
        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        return offset_matrix @ transform_matrix

    def compute_world_matrix(self, parent_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """compute world transform by combining with parent's"""
        local = self.compute_local_matrix()
        if parent_matrix is not None:
            return parent_matrix @ local
        return local

    def compute_transformed_aabb(self) -> Size:
        """compute axis-aligned bounding box (width/height) after local transform"""
        matrix = self.compute_local_matrix()
        corners = np.array(
            [
                [0, 0, 1],
                [self._dimensions.width, 0, 1],
                [0, self._dimensions.height, 1],
                [self._dimensions.width, self._dimensions.height, 1],
            ]
        )
        transformed = np.dot(matrix, corners.T).T
        min_x = np.min(transformed[:, 0])
        max_x = np.max(transformed[:, 0])
        min_y = np.min(transformed[:, 1])
        max_y = np.max(transformed[:, 1])
        return Size(width=max_x - min_x, height=max_y - min_y)

    def get_local_bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in parent coords"""
        matrix = self.compute_local_matrix()
        corners = np.array(
            [
                [0, 0, 1],
                [self._dimensions.width, 0, 1],
                [0, self._dimensions.height, 1],
                [self._dimensions.width, self._dimensions.height, 1],
            ]
        )
        transformed = np.dot(matrix, corners.T).T
        min_x = np.min(transformed[:, 0])
        max_x = np.max(transformed[:, 0])
        min_y = np.min(transformed[:, 1])
        max_y = np.max(transformed[:, 1])
        return (min_x, min_y, max_x, max_y)

    def measure(self, renderer=None) -> Size:
        """measure intrinsic size if needed"""
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def apply_layout(self):
        """apply layout to position children (no-op for base)"""
        pass

    def measure_and_layout(self, renderer=None) -> Size:
        """unified method for measurement and layout"""
        self.measure(renderer)
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render this component (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement render")

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> Dict[str, Any]:
        return self.renderer_options.get(renderer_name, {})
