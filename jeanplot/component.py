# File: jeanplot/component.py
# -*- coding: utf-8 -*-
"""Base class for all visual elements in the scene graph."""

from typing import Optional, Dict, Any, Union, Annotated, List, Tuple, TYPE_CHECKING, Sequence
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

# use absolute imports
from jeanplot.models import Transform, Size, BoxStyle, Offset
from jeanplot.debug import debug_print
from jeanplot.style import jstyle

if TYPE_CHECKING:
    from jeanplot.renderer import BaseRenderer
    from jeanplot.container import Container  # forward reference for parent typing

logger = logging.getLogger(__name__)


def size_from_sequence(seq: Union[tuple, list, Size]) -> Size:
    return seq if isinstance(seq, Size) else Size(width=seq[0], height=seq[1])


ValidatedSize = Annotated[Size, BeforeValidator(size_from_sequence)]


class Component(BaseModel):
    """Base visual element with position, size, style, and transformation."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # --- Configuration ---
    id: Optional[str] = None
    style_class: list[str] = Field(default_factory=list)
    show: bool = True
    debug: bool = False
    # determines if component participates in parent's layout flow
    is_overlay: bool = False

    # --- Geometry & Position ---
    min_dimensions: ValidatedSize = Field(default_factory=Size)
    max_dimensions: ValidatedSize = Field(
        default_factory=partial(Size, width=float("inf"), height=float("inf"))
    )
    # transformation applied *after* offset
    transform: Transform = Field(default_factory=Transform)
    # user-defined offset relative to layout/attachment point
    offset: Offset = Field(default_factory=Offset)
    # attach to another component instead of parent layout flow
    attached_to: Optional[Union[str, "Component"]] = None
    # offset relative to attachment target's origin
    attachment_offset: Offset = Field(default_factory=Offset)
    # list of anchor points, typically added via add_child
    anchor_points: List["Component"] = Field(default_factory=list)

    # --- Style ---
    style: BoxStyle = Field(default_factory=BoxStyle)

    # --- Backend Specific ---
    renderer_options: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))

    # --- Internal State ---
    parent: Optional["Component"] = None
    # calculated size after measurement and constraints
    _dimensions: Size = PrivateAttr(default_factory=Size)
    # size calculated by _measure_natural before constraints
    _natural_dimensions: Size = PrivateAttr(default_factory=Size)
    # position assigned by parent's layout (relative to parent content origin)
    _layout_origin_in_parent: Tuple[float, float] = PrivateAttr(default=(0.0, 0.0))
    # resolved component if attached_to is a string
    _resolved_attach_target: Optional["Component"] = PrivateAttr(default=None)

    def _log_debug(self, message: str, data: Any = None):
        """utility for logging debug messages with component context."""
        comp_id = self.id or self.__class__.__name__
        debug_print(comp_id, message, data)

    @model_validator(mode="after")
    def _validate_dimension_constraints(self):
        """ensure min dimensions are not greater than max dimensions."""
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
        comp_id_str = self.id or self.__class__.__name__

        # --- Exit early if already resolved ---
        if self._resolved_attach_target:
            self._log_debug(
                f"_resolve_attachment: skip (already resolved to '{self._resolved_attach_target.id}')"
            )
            return

        # --- Handle direct component reference ---
        if isinstance(self.attached_to, Component):
            self._resolved_attach_target = self.attached_to
            attach_target_id = self.attached_to.id or type(self.attached_to).__name__
            self._log_debug(
                f"_resolve_attachment: assigned direct component target '{attach_target_id}'"
            )
            return  # Successfully resolved

        # --- Handle string path reference ---
        if isinstance(self.attached_to, str):
            path_to_resolve = self.attached_to
            self._log_debug(
                f"_resolve_attachment: trying to resolve string path '{path_to_resolve}'"
            )
            if not self.parent:
                self._log_debug(
                    f"warning: cannot resolve string path '{path_to_resolve}' without parent link."
                )
                return  # Cannot resolve

            from jeanplot.path_utils import find_component_by_path  # local import

            try:
                root = self
                while root.parent is not None:
                    root = root.parent
                resolved = find_component_by_path(root, path_to_resolve)
                if resolved:
                    self._resolved_attach_target = resolved
                    self._log_debug(
                        f"_resolve_attachment: success string path '{path_to_resolve}' -> '{resolved.id or type(resolved).__name__}'"
                    )
                # else: find_component_by_path raises error if not found
            except ValueError as e:
                self._log_debug(f"_resolve_attachment: string resolution error: {e}")
            return  # Attempted resolution (success or fail)

        # --- Handle None or invalid type ---
        # self._log_debug(f"_resolve_attachment: skip (attached_to is None or invalid type {type(self.attached_to)})")

    def compute_local_matrix(self) -> np.ndarray:
        """
        calculates the 3x3 transformation matrix relative to the component's
        reference point (parent content origin or attachment target origin).
        this includes layout position, user offset, and local transform.
        """
        # 1. start with identity
        matrix = np.identity(3)

        # 2. apply layout position (set by parent) and user offset
        base_ox, base_oy = self._layout_origin_in_parent
        offset_dx, offset_dy = self.offset.compute(
            self._dimensions, self.parent._dimensions if self.parent else None
        )
        ox = base_ox + offset_dx
        oy = base_oy + offset_dy
        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        matrix = offset_matrix @ matrix

        # 3. apply local transform (scale, rotate, skew, translate)
        transform_matrix = self.transform.to_matrix(self._dimensions)
        matrix = transform_matrix @ matrix  # transform applied *after* offset

        if logger.isEnabledFor(logging.DEBUG):  # can be noisy
            self._log_debug(
                f"local matrix: layout=({base_ox:.1f},{base_oy:.1f}), user_delta=({offset_dx:.1f},{offset_dy:.1f}) -> final_offset=({ox:.1f},{oy:.1f})",
                matrix,
            )
        return matrix

    def compute_world_matrix(self, parent_world_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the final 3x3 transformation matrix from the component's
        local coordinate system to world coordinates.
        Handles standard layout hierarchy and attachment.
        """
        comp_id_str = self.id or self.__class__.__name__
        self._log_debug(f"compute_world_matrix: --- start ({comp_id_str}) ---")

        # 1. Resolve attachment reference (string or component)
        self._resolve_attachment()
        target_info = (
            f"'{self._resolved_attach_target.id}'" if self._resolved_attach_target else "None"
        )
        self._log_debug(f"compute_world_matrix: resolved target = {target_info}")

        # 2. Calculate the component's intrinsic transform (applied relative to its final origin)
        intrinsic_transform_matrix = self.transform.to_matrix(self._dimensions)
        # self._log_debug("compute_world_matrix: intrinsic transform matrix", intrinsic_transform_matrix) # Can be noisy

        # 3. Determine the positioning matrix (world coords of component's local origin)
        position_matrix: np.ndarray

        if self._resolved_attach_target:
            self._log_debug("compute_world_matrix: using ATTACHED path")
            self._layout_origin_in_parent = (
                0.0,
                0.0,
            )  # Ensure layout origin is ignored for attached

            target_world_matrix = self._resolved_attach_target.compute_world_matrix()
            self._log_debug("compute_world_matrix: Target world matrix", target_world_matrix)

            attach_ox, attach_oy = self.attachment_offset.compute(
                self._dimensions, self._resolved_attach_target._dimensions
            )
            self._log_debug(
                f"compute_world_matrix: Attachment offset delta = ({attach_ox:.1f}, {attach_oy:.1f})"
            )
            attachment_offset_matrix = np.array([[1, 0, attach_ox], [0, 1, attach_oy], [0, 0, 1]])

            position_matrix = target_world_matrix @ attachment_offset_matrix
            self._log_debug(
                "compute_world_matrix: Position matrix after attachment offset", position_matrix
            )

            # apply user offset relative to the attachment point
            user_offset_dx, user_offset_dy = self.offset.compute(
                self._dimensions, self._resolved_attach_target._dimensions
            )
            self._log_debug(
                f"compute_world_matrix: User offset delta (attach context) = ({user_offset_dx:.1f}, {user_offset_dy:.1f})"
            )
            if user_offset_dx != 0.0 or user_offset_dy != 0.0:
                user_offset_matrix = np.array(
                    [[1, 0, user_offset_dx], [0, 1, user_offset_dy], [0, 0, 1]]
                )
                position_matrix = position_matrix @ user_offset_matrix
                self._log_debug(
                    "compute_world_matrix: Position matrix after user offset (attach context)",
                    position_matrix,
                )

        elif self.parent:
            self._log_debug("compute_world_matrix: using PARENT path")
            parent_world = (
                parent_world_matrix
                if parent_world_matrix is not None
                else self.parent.compute_world_matrix()
            )
            self._log_debug("compute_world_matrix: Parent world matrix", parent_world)

            layout_ox, layout_oy = self._layout_origin_in_parent  # Uses layout position
            self._log_debug(
                f"compute_world_matrix: Layout origin in parent = ({layout_ox:.1f}, {layout_oy:.1f})"
            )

            user_offset_dx, user_offset_dy = self.offset.compute(
                self._dimensions, self.parent._dimensions
            )
            self._log_debug(
                f"compute_world_matrix: User offset delta = ({user_offset_dx:.1f}, {user_offset_dy:.1f})"
            )

            total_offset_x = layout_ox + user_offset_dx
            total_offset_y = layout_oy + user_offset_dy
            self._log_debug(
                f"compute_world_matrix: Total local offset = ({total_offset_x:.1f}, {total_offset_y:.1f})"
            )
            local_positioning_matrix = np.array(
                [[1, 0, total_offset_x], [0, 1, total_offset_y], [0, 0, 1]]
            )
            position_matrix = parent_world @ local_positioning_matrix
            self._log_debug(
                "compute_world_matrix: Position matrix (parent context)", position_matrix
            )
        else:
            self._log_debug("compute_world_matrix: using ROOT path")
            # --- Root Component ---
            user_offset_dx, user_offset_dy = self.offset.compute(self._dimensions, None)
            self._log_debug(
                f"compute_world_matrix: User offset delta (root) = ({user_offset_dx:.1f}, {user_offset_dy:.1f})"
            )
            position_matrix = np.array([[1, 0, user_offset_dx], [0, 1, user_offset_dy], [0, 0, 1]])
            self._log_debug("compute_world_matrix: Position matrix (root context)", position_matrix)

        # 4. Final world matrix = Position @ Intrinsic Transform
        world_matrix = position_matrix @ intrinsic_transform_matrix
        self._log_debug(
            f"compute_world_matrix: --- final world matrix ({comp_id_str}) ---", world_matrix
        )
        return world_matrix

    def get_world_origin(self) -> np.ndarray:
        """calculates the component's origin (top-left) in world coordinates."""
        comp_id_str = self.id or self.__class__.__name__
        self._log_debug(f"get_world_origin: --- start ({comp_id_str}) ---")
        world_matrix = self.compute_world_matrix()
        self._log_debug(
            f"get_world_origin: Matrix used for origin extraction ({comp_id_str})", world_matrix
        )
        origin_world = world_matrix @ np.array([0, 0, 1])
        origin_xy = origin_world[:2]  # return only x, y
        self._log_debug(f"get_world_origin: Extracted origin point -> {origin_xy}")
        return origin_xy

    def get_world_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_x, min_y, max_x, max_y) in world coordinates."""
        if self._dimensions.width <= 0 or self._dimensions.height <= 0:
            # handle zero size: bounds are just the origin point
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
            return None  # indicate failure

    def _apply_style(self):
        """apply styles from the global jstyle object."""
        jstyle.apply(self)

    def _measure_natural(self, renderer: Optional["BaseRenderer"]) -> Size:
        """
        calculates the component's intrinsic ("natural") size based on its
        content, before constraints are applied. must be implemented by subclasses
        that have content (e.g., Text, Container, SVGElement). base implementation
        returns zero size.
        """
        return Size(0, 0)

    def _apply_constraints(self, natural_size: Size) -> Size:
        """applies min/max dimension constraints to a given natural size."""
        constrained_w = min(
            max(self.min_dimensions.width, natural_size.width), self.max_dimensions.width
        )
        constrained_h = min(
            max(self.min_dimensions.height, natural_size.height), self.max_dimensions.height
        )
        final_size = Size(width=max(0, constrained_w), height=max(0, constrained_h))

        if final_size.width != natural_size.width or final_size.height != natural_size.height:
            self._log_debug(f"natural size {natural_size} constrained to {final_size}")

        return final_size

    def _layout_children(self, renderer: Optional["BaseRenderer"]):
        """
        positions and potentially resizes child components. base implementation
        does nothing. container subclasses override this.
        """
        pass  # non-containers have no children layout logic

    def measure_and_layout(self, renderer: Optional["BaseRenderer"] = None):
        """
        coordinates the measurement and layout process for this component.
        """
        comp_id_str = self.id or self.__class__.__name__
        self._log_debug(f"measure_and_layout: --- start ({comp_id_str}) ---")

        # 1. apply styles and resolve attachments first
        self._log_debug("measure_and_layout: applying style...")
        self._apply_style()
        self._log_debug("measure_and_layout: resolving attachment...")
        self._resolve_attachment()

        # 2. measure natural size (implemented by subclasses)
        self._log_debug("measure_and_layout: measuring natural size...")
        self._natural_dimensions = self._measure_natural(renderer)
        self._log_debug(f"measure_and_layout: natural size = {self._natural_dimensions}")

        # 3. apply constraints to get final dimensions
        self._log_debug("measure_and_layout: applying constraints...")
        self._dimensions = self._apply_constraints(self._natural_dimensions)
        self._log_debug(f"measure_and_layout: constrained size = {self._dimensions}")

        # 4. perform internal layout (implemented by subclasses like Container)
        self._log_debug("measure_and_layout: performing internal layout (if applicable)...")
        self._layout_children(renderer)

        self._log_debug(
            f"measure_and_layout: --- finished ({comp_id_str}), final dims: {self._dimensions} ---"
        )

    def render(self, renderer: "BaseRenderer", context: Any, matrix: np.ndarray):
        """
        renders the component using the provided renderer and context.
        subclasses override this to draw specific content. base implementation
        only draws debug visuals if enabled.
        """
        if not self.show:
            return

        # base render is often noisy if logged extensively
        # self._log_debug("render (base)")

        if self.debug:
            if self._dimensions.width <= 0 or self._dimensions.height <= 0:
                self._log_debug("cannot render debug box for zero-size component.")
            else:
                renderer.render_debug(context, self, matrix)

    def add_renderer_option(self, renderer_name: str, option_name: str, value: Any):
        """store renderer-specific options."""
        self.renderer_options[renderer_name][option_name] = value

    def get_renderer_options(self, renderer_name: str) -> Dict[str, Any]:
        """retrieve renderer-specific options."""
        return self.renderer_options.get(renderer_name, {})

    def add_child(self, child: "Component"):
        """adds a child component (primarily for containers)."""
        # base component doesn't store children directly, but handles anchors
        if isinstance(child, AnchorComponent):
            if child not in self.anchor_points:
                self.anchor_points.append(child)
                child.parent = self  # set parent link for anchors too

    def add_children(self, children: Sequence["Component"]):
        """adds multiple children."""
        for child in children:
            self.add_child(child)


class Overlay(Component):
    """component that doesn't participate in parent's layout flow."""

    is_overlay: bool = True


class AnchorComponent(Component):
    """small, often invisible, component used as a connection point."""

    show: bool = False  # typically not rendered visually
    # default direction vector (normalized) for connection routing hints
    direction: Optional[Tuple[float, float]] = None
    # minimum length of the connection segment starting/ending here
    min_segment: float = 10.0
    # force near-zero size for anchors
    min_dimensions: Size = Field(default_factory=lambda: Size(width=1e-6, height=1e-6))
    max_dimensions: Size = Field(default_factory=lambda: Size(width=1e-6, height=1e-6))
    # anchors usually act as overlays unless specifically placed in layout
    is_overlay: bool = True

    @model_validator(mode="after")
    def _normalize_direction_vector(self):
        """ensure direction vector is normalized."""
        if self.direction is not None:
            dx, dy = self.direction
            length = math.hypot(dx, dy)
            if length > 1e-6:
                norm_dir = (dx / length, dy / length)
                if not np.allclose(self.direction, norm_dir):
                    # use object.__setattr__ to bypass validation cycle if needed, though simple assignment should work here
                    self.direction = norm_dir
            else:
                # default direction if input is zero vector (e.g., up)
                self.direction = (0.0, 1.0)
        return self
