from __future__ import annotations

from typing import Any, Annotated, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from jeanplot.core.renderer import BaseRenderer
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    PrivateAttr,
    ConfigDict,
    BeforeValidator,
)
import numpy as np
from collections import defaultdict
from functools import partial
import math
import logging

from jeanplot.core.models import Transform, Size, BoxStyle, Offset
from jeanplot.core.debug import DebugMixin
from jeanplot.core.style import jstyle


logger = logging.getLogger(__name__)


def size_from_sequence(seq: tuple | list | Size) -> Size:
    return seq if isinstance(seq, Size) else Size(width=seq[0], height=seq[1])


ValidatedSize = Annotated[Size, BeforeValidator(size_from_sequence)]


class Component(DebugMixin, BaseModel):
    """base visual element with position, size, style, and transformation."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    id: str | None = None
    style_class: list[str] = Field(default_factory=list)
    show: bool = True
    debug: bool = False
    is_overlay: bool = False
    z_index: int = 0

    min_dimensions: ValidatedSize = Field(default_factory=Size)
    max_dimensions: ValidatedSize = Field(
        default_factory=partial(Size, width=float("inf"), height=float("inf"))
    )
    transform: Transform = Field(default_factory=Transform)
    offset: Offset = Field(default_factory=Offset)
    attached_to: str | Component | None = None
    attachment_offset: Offset = Field(default_factory=Offset)
    anchor_points: list[AnchorComponent] = Field(default_factory=list)

    style: BoxStyle = Field(default_factory=BoxStyle)
    opacity: float = 1.0

    renderer_options: dict[str, dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))

    parent: Component | None = None
    _dimensions: Size = PrivateAttr(default_factory=Size)
    _natural_dimensions: Size = PrivateAttr(default_factory=Size)
    _layout_origin_in_parent: tuple[float, float] = PrivateAttr(default=(0.0, 0.0))
    _resolved_attach_target: Component | None = PrivateAttr(default=None)
    _user_set_fields: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, _context):
        object.__setattr__(self, "_user_set_fields", set(self.model_fields_set))

    @model_validator(mode="after")
    def _validate_dimension_constraints(self):
        if self.min_dimensions.width > self.max_dimensions.width:
            self._log_debug(
                f"warning: min_width > max_width, adjusting max_width to {self.min_dimensions.width}"
            )
            self.max_dimensions.width = self.min_dimensions.width
        if self.min_dimensions.height > self.max_dimensions.height:
            self._log_debug(
                f"warning: min_height > max_height, adjusting max_height to {self.min_dimensions.height}"
            )
            self.max_dimensions.height = self.min_dimensions.height
        return self

    def _resolve_attachment(self):
        """resolve string path or direct reference in attached_to."""
        if self._resolved_attach_target:
            return

        if isinstance(self.attached_to, Component):
            self._resolved_attach_target = self.attached_to
            return

        if isinstance(self.attached_to, str):
            if not self.parent:
                self._log_debug(
                    f"warning: cannot resolve string path '{self.attached_to}' without parent."
                )
                return

            from jeanplot.core.path_utils import find_component_by_path

            try:
                root = self
                while root.parent is not None:
                    root = root.parent
                self._resolved_attach_target = find_component_by_path(root, self.attached_to)
            except ValueError as e:
                self._log_debug(f"_resolve_attachment: string resolution error: {e}")

    def compute_local_matrix(self) -> np.ndarray:
        """3x3 matrix relative to the component's reference point."""
        matrix = np.identity(3)
        base_ox, base_oy = self._layout_origin_in_parent
        parent_dims = self.parent._dimensions if self.parent else None
        offset_dx, offset_dy = self.offset.compute(self._dimensions, parent_dims)
        ox, oy = base_ox + offset_dx, base_oy + offset_dy
        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        matrix = offset_matrix @ matrix
        transform_matrix = self.transform.to_matrix(self._dimensions)
        matrix = transform_matrix @ matrix
        return matrix

    def compute_world_matrix(self, parent_world_matrix: np.ndarray | None = None) -> np.ndarray:
        """final 3x3 matrix to world coordinates."""
        self._resolve_attachment()
        intrinsic_transform_matrix = self.transform.to_matrix(self._dimensions)
        position_matrix: np.ndarray

        if self._resolved_attach_target:
            self._layout_origin_in_parent = (0.0, 0.0)
            target_world_matrix = self._resolved_attach_target.compute_world_matrix()
            attach_ox, attach_oy = self.attachment_offset.compute(
                self._dimensions, self._resolved_attach_target._dimensions
            )
            attachment_offset_matrix = np.array([[1, 0, attach_ox], [0, 1, attach_oy], [0, 0, 1]])
            position_matrix = target_world_matrix @ attachment_offset_matrix
            user_offset_dx, user_offset_dy = self.offset.compute(
                self._dimensions, self._resolved_attach_target._dimensions
            )
            if user_offset_dx != 0.0 or user_offset_dy != 0.0:
                user_offset_matrix = np.array(
                    [[1, 0, user_offset_dx], [0, 1, user_offset_dy], [0, 0, 1]]
                )
                position_matrix = position_matrix @ user_offset_matrix
        elif self.parent:
            parent_world = (
                parent_world_matrix
                if parent_world_matrix is not None
                else self.parent.compute_world_matrix()
            )
            layout_ox, layout_oy = self._layout_origin_in_parent
            user_offset_dx, user_offset_dy = self.offset.compute(
                self._dimensions, self.parent._dimensions
            )
            total_offset_x, total_offset_y = layout_ox + user_offset_dx, layout_oy + user_offset_dy
            local_positioning_matrix = np.array(
                [[1, 0, total_offset_x], [0, 1, total_offset_y], [0, 0, 1]]
            )
            position_matrix = parent_world @ local_positioning_matrix
        else:
            user_offset_dx, user_offset_dy = self.offset.compute(self._dimensions, None)
            position_matrix = np.array([[1, 0, user_offset_dx], [0, 1, user_offset_dy], [0, 0, 1]])

        world_matrix = position_matrix @ intrinsic_transform_matrix
        return world_matrix

    def get_world_origin(self) -> np.ndarray:
        """origin (top-left) in world coordinates."""
        world_matrix = self.compute_world_matrix()
        origin_world = world_matrix @ np.array([0, 0, 1])
        return origin_world[:2]

    def get_world_bounds(self) -> tuple[float, float, float, float] | None:
        """(min_x, min_y, max_x, max_y) in world coordinates."""
        if self._dimensions.width <= 0 or self._dimensions.height <= 0:
            ox, oy = self.get_world_origin()
            return (ox, oy, ox, oy)

        w, h = self._dimensions.width, self._dimensions.height
        corners_local = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
        world_matrix = self.compute_world_matrix()
        world_corners = (world_matrix @ corners_local).T

        if world_corners.shape[0] > 0:
            min_x, min_y = np.min(world_corners[:, 0]), np.min(world_corners[:, 1])
            max_x, max_y = np.max(world_corners[:, 0]), np.max(world_corners[:, 1])
            return (min_x, min_y, max_x, max_y)
        else:
            self._log_debug("warning: failed to transform corners for world bounds.")
            return None

    @property
    def safe_style(self) -> BoxStyle:
        return self.style or BoxStyle()

    def _apply_style(self):
        """apply styles from the global jstyle object. children are styled during their own pass."""
        jstyle.apply_one(self)

    def _measure_natural(self, renderer: "BaseRenderer | None") -> Size:
        """intrinsic size. subclasses override."""
        return Size(0, 0)

    def _apply_constraints(self, natural_size: Size) -> Size:
        constrained_w = min(
            max(self.min_dimensions.width, natural_size.width), self.max_dimensions.width
        )
        constrained_h = min(
            max(self.min_dimensions.height, natural_size.height), self.max_dimensions.height
        )
        final_size = Size(width=max(0, constrained_w), height=max(0, constrained_h))
        return final_size

    def _layout_children(self, renderer: "BaseRenderer | None"):
        """positions child components. container subclasses override."""
        pass

    def measure_and_layout(self, renderer: "BaseRenderer | None" = None):
        self._apply_style()
        self._resolve_attachment()
        self._natural_dimensions = self._measure_natural(renderer)
        self._dimensions = self._apply_constraints(self._natural_dimensions)
        self._layout_children(renderer)

    def render(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        """subclasses override."""
        if not self.show:
            return
        if self.debug:
            if self._dimensions.width <= 0 or self._dimensions.height <= 0:
                self._log_debug("cannot render debug box for zero-size component.")
            else:
                renderer.render_debug(context, self, matrix)

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> dict[str, Any]:
        return self.renderer_options.get(renderer_name, {})

    def add_child(self, child: Component):
        if isinstance(child, AnchorComponent):
            if child not in self.anchor_points:
                self.anchor_points.append(child)
            child.parent = self

    def add_children(self, children: Sequence[Component]):
        for child in children:
            self.add_child(child)


class Overlay(Component):
    """component that doesn't participate in parent's layout flow."""

    is_overlay: bool = True


class AnchorComponent(Component):
    """small, often invisible, component used as a connection point."""

    show: bool = False
    direction: tuple[float, float] | None = None
    min_segment: float = 10.0
    min_dimensions: Size = Field(default_factory=lambda: Size(width=1e-6, height=1e-6))
    max_dimensions: Size = Field(default_factory=lambda: Size(width=1e-6, height=1e-6))
    is_overlay: bool = True

    @model_validator(mode="after")
    def _normalize_direction_vector(self):
        if self.direction is not None:
            dx, dy = self.direction
            length = math.hypot(dx, dy)
            if length > 1e-6:
                norm_dir = (dx / length, dy / length)
                if not np.allclose(self.direction, norm_dir):
                    self.direction = norm_dir
            else:
                self.direction = (0.0, 1.0)
        return self


# resolve forward references for direct-module imports
Component.model_rebuild(force=True)
AnchorComponent.model_rebuild(force=True)
Overlay.model_rebuild(force=True)
