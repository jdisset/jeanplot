from typing import Optional, Dict, Any, Union, Annotated, List, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator, PrivateAttr, BeforeValidator, ConfigDict
import numpy as np
from collections import defaultdict
from functools import partial
import math

from .models import Transform, Size, BoxStyle, Offset, LayoutConstraints
from .debug import debug_print
from .style import jstyle


def size_from_sequence(seq: Union[tuple, list, Size]) -> Size:
    return seq if isinstance(seq, Size) else Size(width=seq[0], height=seq[1])


ValidatedSize = Annotated[Size, BeforeValidator(size_from_sequence)]


class Component(BaseModel):
    """base component class - anything that can be rendered"""

    model_config = ConfigDict(validate_assignment=True)

    id: Optional[str] = None
    transform: Transform = Field(default_factory=Transform)
    offset: Offset = Field(default_factory=Offset)
    attached_to: Optional[Union[str, "Component"]] = None
    attachment_offset: Offset = Field(default_factory=Offset)
    min_dimensions: ValidatedSize = Field(default_factory=Size)
    max_dimensions: ValidatedSize = Field(
        default_factory=partial(Size, width=float("inf"), height=float("inf"))
    )
    style: Any = Field(default_factory=BoxStyle)
    style_class: list[str] = []
    show: bool = True
    is_overlay: bool = False
    renderer_options: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))
    debug: bool = False
    parent: Optional["Component"] = None
    anchor_points: List["Component"] = Field(default_factory=list)

    _dimensions: Size = PrivateAttr(default_factory=Size)
    _transformed_aabb: Size = PrivateAttr(default_factory=Size)
    _resolved_attach_target: Optional["Component"] = PrivateAttr(default=None)

    def _get_path(self):
        path = self.id or self.__class__.__name__
        return f"{self.parent._get_path()}/{path}" if self.parent else path

    def _log_debug(self, message: str, data=None):
        if self.debug:
            debug_print(self.id or "Component", message, data)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.min_dimensions.width > self.max_dimensions.width:
            raise ValueError("min>max width")
        if self.min_dimensions.height > self.max_dimensions.height:
            raise ValueError("min>max height")
        return self

    def model_post_init(self, __context: Any) -> None:
        pass

    def compute_local_matrix(self) -> np.ndarray:
        """convert local transform to matrix, including layout offset"""
        # Ensure dimensions are available, default to 0,0 if not measured
        if not hasattr(self, "_dimensions"):
            # self._log_debug(f"Warning: compute_local_matrix called before measurement for {self.id}. Using default Size(0,0).")
            self._dimensions = Size()

        # Calculate local transformations (scale, rotate, skew) first
        transform_matrix = self.transform.to_matrix(self._dimensions)

        # Calculate the offset relative to the parent origin
        parent_dims = getattr(self.parent, "_dimensions", None) if self.parent else None
        # Use max(0, ...) to avoid potential negative dimensions if something went wrong
        self_width = max(0, self._dimensions.width)
        self_height = max(0, self._dimensions.height)
        parent_width = max(0, parent_dims.width) if parent_dims else 0
        parent_height = max(0, parent_dims.height) if parent_dims else 0

        ox, oy = self.offset.compute(
            Size(width=self_width, height=self_height),
            Size(width=parent_width, height=parent_height),
        )

        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        final_local_matrix = offset_matrix @ transform_matrix

        # --- Refined Debug Print ---
        if getattr(self, "debug", False):
            # Use self._log_debug for consistency if available, otherwise print
            log_func = getattr(
                self,
                "_log_debug",
                lambda msg, data=None: print(
                    f"DEBUG [{self.id or 'Component'}] {msg}"
                    + (f": {data}" if data is not None else "")
                ),
            )
            log_func(f"compute_local_matrix ({self.id}):")
            log_func(f"  Offset Compute -> ({ox:.3f}, {oy:.3f}) for offset={self.offset}")
            log_func(f"  Offset Matrix:\n{np.round(offset_matrix, 3)}")
            log_func(f"  Transform Matrix:\n{np.round(transform_matrix, 3)}")
            log_func(
                f"  Final Local Matrix (Offset * Transform):\n{np.round(final_local_matrix, 3)}"
            )
        # --- End Debug Print ---

        return final_local_matrix

    def compute_world_matrix(self, parent_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """compute world transform, considering attachment or standard layout"""
        # --- Add Debug ---
        debug_enabled = getattr(self, "debug", False)
        if hasattr(self, "_log_debug"):
            log_func = self._log_debug
        else:
            log_func = lambda msg, data=None: print(
                f"DEBUG [{getattr(self, 'id', 'Component')}] {msg}"
                + (f": {data}" if data is not None else "")
            )

        if debug_enabled:
            log_func(f"compute_world_matrix ({getattr(self, 'id', 'N/A')}) called")
        # --- End Debug ---

        attach_target = self._resolved_attach_target or (
            self.attached_to if isinstance(self.attached_to, Component) else None
        )

        final_world_matrix: np.ndarray  # Declare type hint

        if attach_target:
            # --- Attachment Path (Simplified) ---
            if debug_enabled:
                log_func(f"  Using attachment path for {getattr(self, 'id', 'N/A')}")
            target_matrix = attach_target.compute_world_matrix()  # Recursive call
            local_transform = self.transform.to_matrix(self._dimensions)
            target_dims = getattr(attach_target, "_dimensions", Size())

            if not hasattr(self, "_dimensions"):
                self.measure_and_layout()  # Ensure self is measured

            ox, oy = self.attachment_offset.compute(self._dimensions, target_dims)
            attachment_offset_local_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
            final_world_matrix = target_matrix @ attachment_offset_local_matrix @ local_transform
            if debug_enabled:
                log_func(f"  Attachment Target Matrix:\n{np.round(target_matrix, 3)}")
                log_func(
                    f"  Attachment Offset Matrix:\n{np.round(attachment_offset_local_matrix, 3)}"
                )
                log_func(f"  Local Transform Matrix:\n{np.round(local_transform, 3)}")
        else:
            # --- Standard Layout Path ---
            if debug_enabled:
                log_func(f"  Using standard layout path for {getattr(self, 'id', 'N/A')}")

            # Calculate local matrix (logs internally if debug=True)
            local_matrix = self.compute_local_matrix()

            # Determine effective parent matrix
            effective_parent_matrix: Optional[np.ndarray] = None
            if parent_matrix is not None:
                effective_parent_matrix = parent_matrix
                if debug_enabled:
                    log_func(f"  Using parent matrix override")
            elif self.parent is not None:
                effective_parent_matrix = self.parent.compute_world_matrix()
                if debug_enabled:
                    log_func(f"  Got parent matrix from self.parent")
            else:
                if debug_enabled:
                    log_func(f"  No parent matrix (root)")

            # Calculate final world matrix
            if effective_parent_matrix is not None:
                final_world_matrix = effective_parent_matrix @ local_matrix
            else:  # Root component case
                final_world_matrix = local_matrix

            # --- Debug for Standard Path ---
            if debug_enabled:
                log_func(
                    f"  Local Matrix (from compute_local_matrix):\n{np.round(local_matrix, 3)}"
                )
                if effective_parent_matrix is not None:
                    log_func(f"  Effective Parent Matrix:\n{np.round(effective_parent_matrix, 3)}")
                else:
                    log_func(f"  Effective Parent Matrix: None (Root)")
                log_func(
                    f"  Calculated World Matrix (Parent * Local):\n{np.round(final_world_matrix, 3)}"
                )
            # --- End Debug ---

        if debug_enabled:
            log_func(
                f"  compute_world_matrix ({getattr(self, 'id', 'N/A')}) returning:\n{np.round(final_world_matrix, 3)}"
            )

        # Ensure the function *actually* returns the calculated matrix
        return final_world_matrix

    def _transform_points(self, points, matrix):
        transformed = (matrix @ points.T).T

        if not np.all(np.isfinite(transformed)):
            self._log_debug(f"Warning: Non-finite values in transformed points for {self.id}.")
            finite_points = transformed[np.all(np.isfinite(transformed), axis=1)]
            return finite_points if finite_points.shape[0] >= 2 else None

        return transformed

    def compute_transformed_aabb(self) -> Size:
        """compute axis-aligned bounding box after local transform + layout offset"""
        if not hasattr(self, "_dimensions"):
            self.measure()

        matrix = self.compute_local_matrix()
        w, h = max(0, self._dimensions.width), max(0, self._dimensions.height)

        # Handle zero dimension case
        if w == 0 or h == 0:
            origin = np.array([0, 0, 1])
            corner = np.array([w, h, 1])
            transformed_origin = (matrix @ origin)[:2]
            transformed_corner = (matrix @ corner)[:2]
            width = abs(transformed_corner[0] - transformed_origin[0])
            height = abs(transformed_corner[1] - transformed_origin[1])
            return Size(width=width, height=height)

        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        transformed = self._transform_points(corners, matrix)

        if transformed is None:
            return Size()

        min_x, min_y = np.min(transformed[:, 0]), np.min(transformed[:, 1])
        max_x, max_y = np.max(transformed[:, 0]), np.max(transformed[:, 1])

        return Size(width=max(0, max_x - min_x), height=max(0, max_y - min_y))

    def get_local_bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) relative to parent/attach point origin"""
        if not hasattr(self, "_dimensions"):
            self.measure()

        matrix = self.compute_local_matrix()
        w, h = self._dimensions.width, self._dimensions.height

        if w < 0 or h < 0:
            return (0, 0, 0, 0)

        # Handle zero dimension case
        if w == 0 or h == 0:
            origin = np.array([0, 0, 1])
            corner = np.array([w, h, 1])
            transformed_origin = (matrix @ origin)[:2]
            transformed_corner = (matrix @ corner)[:2]
            min_x = min(transformed_origin[0], transformed_corner[0])
            max_x = max(transformed_origin[0], transformed_corner[0])
            min_y = min(transformed_origin[1], transformed_corner[1])
            max_y = max(transformed_origin[1], transformed_corner[1])
            return (min_x, min_y, max_x, max_y)

        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        transformed = self._transform_points(corners, matrix)

        if transformed is None:
            return (0, 0, 0, 0)

        min_x, min_y = np.min(transformed[:, 0]), np.min(transformed[:, 1])
        max_x, max_y = np.max(transformed[:, 0]), np.max(transformed[:, 1])

        return (min_x, min_y, max_x, max_y)

    def measure(self, renderer=None) -> Size:
        if not hasattr(self, "_dimensions"):
            self._dimensions = Size()

        # Apply constraints
        self._dimensions.width = min(
            max(self.min_dimensions.width, self._dimensions.width), self.max_dimensions.width
        )
        self._dimensions.height = min(
            max(self.min_dimensions.height, self._dimensions.height), self.max_dimensions.height
        )

        self._dimensions.width = max(0, self._dimensions.width)
        self._dimensions.height = max(0, self._dimensions.height)

        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def apply_layout(self):
        pass

    def _resolve_attachment(self):
        if (
            isinstance(self.attached_to, str)
            and self._resolved_attach_target is None
            and self.parent
        ):
            from .path_utils import find_component_by_path

            root = self.parent
            while root.parent is not None:
                root = root.parent

            resolved = find_component_by_path(root, self.attached_to)
            if resolved:
                self._resolved_attach_target = resolved
            else:
                self._log_debug(f"Warning: Failed to resolve attached_to path '{self.attached_to}'")
        elif isinstance(self.attached_to, Component):
            self._resolved_attach_target = self.attached_to

    def measure_and_layout(self, renderer=None) -> Size:
        """unified method for measurement and layout with styling"""
        jstyle.apply(self)
        self._resolve_attachment()
        measured_size = self.measure(renderer)
        self.apply_layout()
        return measured_size

    def render(self, renderer, context, matrix: np.ndarray):
        if not self.show:
            return

        if self.debug:
            if not hasattr(self, "_dimensions"):
                self.measure_and_layout(renderer)

            if self._dimensions.width > 0 or self._dimensions.height > 0:
                renderer.render_debug(context, self, matrix)

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> Dict[str, Any]:
        return self.renderer_options.get(renderer_name, {})

    def find_best_anchor_point(self, other_component) -> Optional["Component"]:
        if not hasattr(self, "_dimensions"):
            self.measure_and_layout()

        if not hasattr(other_component, "_dimensions"):
            other_component.measure_and_layout()

        valid_anchors = [a for a in self.anchor_points if isinstance(a, Component)]
        if not valid_anchors:
            return None

        other_matrix = other_component.compute_world_matrix()
        other_center_local = np.array(
            [other_component._dimensions.width / 2, other_component._dimensions.height / 2, 1]
        )
        other_center_world = (other_matrix @ other_center_local)[:2]

        best_score = float("inf")
        best_anchor = None

        for anchor_comp in valid_anchors:
            if not hasattr(anchor_comp, "_dimensions"):
                anchor_comp.measure_and_layout()

            anchor_pos_world = get_world_origin(anchor_comp)
            dx, dy = (
                other_center_world[0] - anchor_pos_world[0],
                other_center_world[1] - anchor_pos_world[1],
            )
            anchor_dist_sq = dx * dx + dy * dy

            if anchor_dist_sq < best_score:
                best_score = anchor_dist_sq
                best_anchor = anchor_comp

        return best_anchor


class Overlay(Component):
    is_overlay: bool = True


class AnchorComponent(Component):
    show: bool = False
    direction: Tuple[float, float] = (0, 1)
    min_segment: float = 10.0
    min_dimensions: Size = Field(default_factory=lambda: Size(width=1e-3, height=1e-3))
    max_dimensions: Size = Field(default_factory=lambda: Size(width=1e-3, height=1e-3))
    style: BoxStyle = Field(default_factory=BoxStyle)
    is_overlay: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._normalize_direction()

    def _normalize_direction(self):
        dx, dy = self.direction
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0:
            object.__setattr__(self, "direction", (dx / length, dy / length))
        else:
            object.__setattr__(self, "direction", (0, 1))


def get_world_origin(component: Component) -> np.ndarray:
    if not hasattr(component, "_dimensions"):
        component.measure_and_layout()

    world_matrix = component.compute_world_matrix()
    origin_local = np.array([0, 0, 1])
    origin_world = world_matrix @ origin_local
    return origin_world[:2]
