from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import numpy as np
from collections import defaultdict
from functools import partial
import math
from .models import Transform, Size, BoxStyle, Offset, AnchorPoint
from .debug import debug_print
from .style import jstyle


class Component(BaseModel):
    """base component class - anything that can be rendered"""

    id: Optional[str] = None
    transform: Transform = Field(default_factory=Transform)
    offset: Offset = Field(default_factory=Offset)
    anchor_points: list[AnchorPoint] = Field(default_factory=list)

    min_dimensions: Size = Field(default_factory=Size)
    max_dimensions: Size = Field(
        default_factory=partial(Size, width=float("inf"), height=float("inf"))
    )

    style: Any = Field(default_factory=BoxStyle)

    # can be useful for styling selection (similar to css class):
    style_class: list[str] = []

    renderer_options: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))
    debug: bool = False

    is_overlay: bool = False
    parent: Optional["Component"] = None

    _dimensions: Size = PrivateAttr(default_factory=Size)
    _transformed_aabb: Size = PrivateAttr(default_factory=Size)

    @model_validator(mode="after")
    def apply_styles(self):
        """apply global styles after model creation"""
        jstyle.apply(self)
        return self

    def _get_path(self):
        """get component path in hierarchy for styling"""
        path = self.id or self.__class__.__name__
        if self.parent:
            return f"{self.parent._get_path()}/{path}"
        return path

    def _log_debug(self, message: str, data=None):
        """helper to log debug messages with component id"""
        debug_print(self.id or "Component", message, data)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.min_dimensions.width > self.max_dimensions.width:
            raise ValueError("min_dimensions.width cannot exceed max_dimensions.width")
        if self.min_dimensions.height > self.max_dimensions.height:
            raise ValueError("min_dimensions.height cannot exceed max_dimensions.height")
        return self

    def find_best_anchor_point(self, other_component) -> Optional[AnchorPoint]:
        """find best anchor point to connect to another component"""
        if not self.anchor_points:
            return None

        # get world centers
        my_dims = getattr(self, "_dimensions", Size(width=1, height=1))
        my_matrix = self.compute_world_matrix()
        my_center = my_matrix @ np.array([my_dims.width / 2, my_dims.height / 2, 1])

        other_dims = getattr(other_component, "_dimensions", Size(width=1, height=1))
        other_matrix = other_component.compute_world_matrix()
        other_center = other_matrix @ np.array([other_dims.width / 2, other_dims.height / 2, 1])

        # ideal direction would point from my center toward other center
        dx = other_center[0] - my_center[0]
        dy = other_center[1] - my_center[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 0:
            ideal_dir = (dx / dist, dy / dist)
        else:
            ideal_dir = (0, 1)

        # score each anchor by alignment with ideal direction and distance
        best_score = float("-inf")
        best_anchor = None

        for anchor in self.anchor_points:
            anchor_pos = anchor.get_world_position(self)

            # calculate distance from anchor to other center
            ax, ay = anchor_pos
            a_dx = other_center[0] - ax
            a_dy = other_center[1] - ay
            anchor_dist = math.sqrt(a_dx * a_dx + a_dy * a_dy)

            # score combines alignment and inverse distance
            # alignment = anchor.direction[0] * ideal_dir[0] + anchor.direction[1] * ideal_dir[1]
            # score = alignment - (anchor_dist / 1000)
            score = anchor_dist

            if score > best_score:
                best_score = score
                best_anchor = anchor

        return best_anchor

    def compute_layout_matrix(self) -> np.ndarray:
        """compute transform matrix for layout only (no offset)"""
        return self.transform.to_matrix(self._dimensions)

    def compute_local_matrix(self) -> np.ndarray:
        """convert local transform to matrix with offset applied after transform"""
        transform_matrix = self.transform.to_matrix(self._dimensions)

        # Get parent dimensions for parent_relative offset
        parent_dims = None
        if self.parent is not None:
            parent_dims = getattr(self.parent, "_dimensions", None)

        ox, oy = self.offset.compute(self._dimensions, parent_dims)
        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        result = offset_matrix @ transform_matrix

        if self.debug:
            self._log_debug(
                "compute_local_matrix",
                {"transform": transform_matrix, "offset": (ox, oy), "result": result},
            )

        return result

    def compute_world_matrix(self, parent_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """compute world transform by combining with parent's"""
        local = self.compute_local_matrix()

        if parent_matrix is None and self.parent is not None:
            parent_matrix = self.parent.compute_world_matrix()

        if parent_matrix is not None:
            result = parent_matrix @ local
        else:
            result = local

        if self.debug:
            self._log_debug(
                "compute_world_matrix",
                {
                    "local": local,
                    "parent_matrix": parent_matrix.tolist()
                    if parent_matrix is not None
                    else "None",
                    "result": result,
                },
            )

        return result

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

        result = Size(width=max_x - min_x, height=max_y - min_y)

        if self.debug:
            self._log_debug(
                "compute_transformed_aabb",
                {
                    "dimensions": (self._dimensions.width, self._dimensions.height),
                    "result": (result.width, result.height),
                    "bounds": (min_x, min_y, max_x, max_y),
                },
            )

        return result

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

        result = (min_x, min_y, max_x, max_y)

        if self.debug:
            self._log_debug("get_local_bounds", {"matrix": matrix, "result": result})

        return result

    def measure(self, renderer=None) -> Size:
        """measure intrinsic size if needed"""
        if self.debug:
            self._log_debug("Measuring component")

        self._transformed_aabb = self.compute_transformed_aabb()

        if self.debug:
            self._log_debug(
                "Measured dimensions",
                {"dimensions": (self._dimensions.width, self._dimensions.height)},
            )

        return self._dimensions

    def apply_layout(self):
        """apply layout to position children (no-op for base)"""
        pass

    def measure_and_layout(self, renderer=None) -> Size:
        """unified method for measurement and layout with styling"""
        if self.debug:
            self._log_debug("measure_and_layout")

        jstyle.apply(self)

        self.measure(renderer)
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render this component (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement render")

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> Dict[str, Any]:
        return self.renderer_options.get(renderer_name, {})
