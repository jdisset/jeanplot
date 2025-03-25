from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import numpy as np
from collections import defaultdict
from .models import Transform, Size, VisualStyle, Offset
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
