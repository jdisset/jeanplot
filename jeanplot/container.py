"""Container component that manages and lays out child components."""

from pydantic import Field, PrivateAttr
from typing import List, Optional, Any, Dict, Tuple, Sequence
import numpy as np
import logging
import inspect

# use absolute imports
from jeanplot.component import Component, AnchorComponent
from jeanplot.models import Size, LayoutConstraints, BoxStyle, Offset
from jeanplot.debug import debug_print
from jeanplot.style import jstyle

# forward ref for type hint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jeanplot.renderer import BaseRenderer

logger = logging.getLogger(__name__)


class Container(Component):
    """component that holds, lays out, and renders child components."""

    children: List[Component] = Field(default_factory=list)
    layout: LayoutConstraints = Field(default_factory=LayoutConstraints)

    _layout_children_cache: List[Component] = PrivateAttr(default_factory=list)
    _overlay_children_cache: List[Component] = PrivateAttr(default_factory=list)

    def add_child(self, child: Component):
        """add child component and set parent link."""
        if child not in self.children:
            self.children.append(child)
        child.parent = self
        # also add anchors to the component's anchor_points list
        if isinstance(child, AnchorComponent) and child not in self.anchor_points:
            self.anchor_points.append(child)

    def add_children(self, children: Sequence[Component]):
        """add multiple children."""
        for child in children:
            self.add_child(child)

    def _prepare_children(self):
        """ensure parent links are set and categorize children for layout/overlay."""
        layout_children = []
        overlay_children = []
        all_children = self.children + [
            anchor for anchor in self.anchor_points if anchor not in self.children
        ]  # ensure anchors are considered

        # process defined anchor points first, ensuring they are added
        for anchor in self.anchor_points:
            if anchor not in self.children:
                self.children.append(anchor)  # add to main list if missing
            anchor.parent = self  # ensure parent link

        # now categorize all children
        for child in self.children:
            if child:
                child.parent = self  # ensure parent link
                if child.is_overlay or child.attached_to:
                    overlay_children.append(child)
                else:
                    layout_children.append(child)

        self._layout_children_cache = layout_children
        self._overlay_children_cache = overlay_children

    def _measure_natural(self, renderer: Optional["BaseRenderer"]) -> Size:
        """calculates natural size based on layout children."""
        self._prepare_children()  # ensure parent links and categorize

        # 1. measure all children recursively first to get their sizes
        # overlays/attached are measured too, as they might have intrinsic size
        for child in self.children:
            if child:
                child.measure_and_layout(renderer)  # measure each child

        # 2. calculate container's natural size based on layout children only
        natural_width, natural_height = 0.0, 0.0
        layout_children = self._layout_children_cache

        if not layout_children:
            insets = self.style.content_inset()
            natural_width = insets[1] + insets[3]  # left + right padding
            natural_height = insets[0] + insets[2]  # top + bottom padding
        else:
            is_row = self.layout.direction == "row"
            main_axis_size, cross_axis_size = 0.0, 0.0
            for i, child in enumerate(layout_children):
                child_natural_size = child._natural_dimensions
                # fallback: use constrained size if natural is zero (e.g., for components sized by min/max)
                child_dims_to_use = child_natural_size
                if child_natural_size.width <= 1e-6 and child_natural_size.height <= 1e-6:
                    child_dims_to_use = child._dimensions  # use the final constrained dims

                self._log_debug(
                    f"  using dims {child_dims_to_use} for child '{child.id or type(child).__name__}' in natural calc"
                )

                margins = child.style.margin  # t, r, b, l
                if is_row:
                    main_axis_size += (
                        child_dims_to_use.width + margins[3] + margins[1]  # w + l + r
                    )
                    cross_axis_size = max(
                        cross_axis_size,
                        child_dims_to_use.height + margins[0] + margins[2],  # h + t + b
                    )
                else:  # column
                    main_axis_size += (
                        child_dims_to_use.height + margins[0] + margins[2]  # h + t + b
                    )
                    cross_axis_size = max(
                        cross_axis_size,
                        child_dims_to_use.width + margins[3] + margins[1],  # w + l + r
                    )
                if i > 0:  # add gap after first child
                    main_axis_size += self.layout.gap

            # assign width/height based on direction
            natural_width, natural_height = (
                (main_axis_size, cross_axis_size) if is_row else (cross_axis_size, main_axis_size)
            )

            # add container's own padding
            insets = self.style.content_inset()
            natural_width += insets[1] + insets[3]
            natural_height += insets[0] + insets[2]

        # ensure non-negative natural size
        natural_width = max(0.0, natural_width)
        natural_height = max(0.0, natural_height)
        self._log_debug(f"_measure_natural result: w={natural_width:.1f}, h={natural_height:.1f}")
        return Size(width=natural_width, height=natural_height)

    def _layout_children(self, renderer: Optional["BaseRenderer"]):
        """positions layout children and triggers layout in child containers."""
        self._prepare_children()  # ensure parent links and categorize

        layout_children = self._layout_children_cache
        if not layout_children:
            # still need to layout overlays if they are containers
            self._layout_overlay_containers(renderer)
            return

        # calculate available content area based on FINAL container dimensions
        content_w, content_h = self.style.content_box(self._dimensions)
        content_x, content_y = self.style.padding[3], self.style.padding[0]  # left, top

        # --- stretch applicable children ---
        # note: stretch happens *before* positioning
        stretched_children = self._calculate_and_apply_stretch(
            content_w, content_h, layout_children
        )

        # --- position layout children ---
        # uses the (potentially stretched) _dimensions of children
        self._position_layout_children(content_x, content_y, content_w, content_h, layout_children)

        # --- recursively layout child containers (including stretched ones) ---
        # this is needed if a child container's size changed due to stretch
        # or if its internal layout depends on its final assigned position (less common)
        for child in layout_children:
            if isinstance(child, Container):
                # re-run internal layout if it was stretched or if its content depends on final size
                if child in stretched_children:
                    self._log_debug(
                        f"re-running layout for stretched child container '{child.id or type(child).__name__}'"
                    )
                    child._layout_children(renderer)
                # simple containers might not need full re-layout, but doesn't hurt
                # child._layout_children(renderer) # simpler version

        # --- Layout overlay containers ---
        # overlays are positioned independently, but if they are containers,
        # their internal children need to be laid out.
        self._layout_overlay_containers(renderer)

    def _layout_overlay_containers(self, renderer: Optional["BaseRenderer"]):
        """triggers layout pass for overlay children that are containers."""
        for child in self._overlay_children_cache:
            if isinstance(child, Container):
                # overlays position themselves via offset/attachment,
                # but need internal layout pass for their own children.
                # their measure_and_layout was already called during the main pass,
                # so their size is known. just run the internal layout part.
                self._log_debug(
                    f"running internal layout for overlay container '{child.id or type(child).__name__}'"
                )
                child._layout_children(renderer)

    def _calculate_and_apply_stretch(
        self, content_w: float, content_h: float, layout_children: List[Component]
    ) -> List[Component]:
        """calculates and applies stretch to children, returning list of stretched children."""
        if self.layout.align_items != "stretch" or not layout_children:
            return []

        stretched_children = []
        is_row = self.layout.direction == "row"
        cross_axis = "height" if is_row else "width"
        main_axis = "width" if is_row else "height"
        available_cross = content_h if is_row else content_w

        for child in layout_children:
            # use child's final constrained dimensions as basis for potential stretch
            current_dims = child._dimensions
            min_dims, max_dims = child.min_dimensions, child.max_dimensions
            margin_t, margin_r, margin_b, margin_l = child.style.margin
            margins_cross = (margin_t + margin_b) if is_row else (margin_l + margin_r)

            target_cross_size = max(0, available_cross - margins_cross)
            min_cross = getattr(min_dims, cross_axis)
            max_cross = getattr(max_dims, cross_axis)

            # only stretch if target size respects min/max
            final_cross_size = current_dims.height  # default to current
            if min_cross <= target_cross_size <= max_cross:
                final_cross_size = target_cross_size
            elif target_cross_size < min_cross:
                final_cross_size = min_cross
            elif target_cross_size > max_cross:
                final_cross_size = max_cross

            # ensure main axis still respects its constraints (use current main axis size)
            min_main = getattr(min_dims, main_axis)
            max_main = getattr(max_dims, main_axis)
            current_main = getattr(current_dims, main_axis)
            final_main_size = min(max(min_main, current_main), max_main)

            # apply changes if needed
            current_cross_val = getattr(current_dims, cross_axis)
            current_main_val = getattr(current_dims, main_axis)

            needs_update = False
            if abs(current_cross_val - final_cross_size) > 1e-6:
                setattr(child._dimensions, cross_axis, final_cross_size)
                needs_update = True
            if abs(current_main_val - final_main_size) > 1e-6:
                setattr(child._dimensions, main_axis, final_main_size)
                needs_update = True

            if needs_update:
                self._log_debug(
                    f"stretched child '{child.id or type(child).__name__}' to {child._dimensions}"
                )
                stretched_children.append(child)

        return stretched_children

    def _position_layout_children(
        self, content_x, content_y, content_w, content_h, layout_children
    ):
        """calculates and sets _layout_origin_in_parent for layout children."""
        num_children = len(layout_children)
        if num_children == 0:
            return

        is_row = self.layout.direction == "row"
        size_attr = "width" if is_row else "height"
        margin_start_idx = 3 if is_row else 0  # left or top
        margin_end_idx = 1 if is_row else 2  # right or bottom
        cross_margin_start_idx = 0 if is_row else 3  # top or left
        cross_margin_end_idx = 2 if is_row else 1  # bottom or right
        cross_size_attr = "height" if is_row else "width"

        # calculate total space needed along the main axis using FINAL child dimensions
        total_child_main_size_with_margins = 0
        for child in layout_children:
            margins = child.style.margin
            total_child_main_size_with_margins += (
                getattr(child._dimensions, size_attr)
                + margins[margin_start_idx]
                + margins[margin_end_idx]
            )
        total_gap = self.layout.gap * max(0, num_children - 1)
        required_space = total_child_main_size_with_margins + total_gap
        available_space = content_w if is_row else content_h
        extra_space = max(0, available_space - required_space)

        # calculate starting position and spacing based on justify_content
        current_pos_main = content_x if is_row else content_y
        justify = self.layout.justify_content
        spacing = 0.0  # spacing between elements (for distribute modes)

        if justify == "center":
            current_pos_main += extra_space / 2.0
        elif justify == "end":
            current_pos_main += extra_space
        elif justify == "space-between" and num_children > 1:
            spacing = extra_space / (num_children - 1)
        elif justify == "space-around":
            spacing = extra_space / num_children if num_children > 0 else 0
            current_pos_main += spacing / 2.0
        elif justify == "space-evenly":
            spacing = extra_space / (num_children + 1) if num_children >= 0 else 0
            current_pos_main += spacing
        # else 'start': current_pos_main is correct

        # position each child
        for i, child in enumerate(layout_children):
            if child.attached_to or child.is_overlay:
                self._log_debug(
                    f"  SKIPPING position calculation for non-layout child '{child.id or type(child).__name__}' (attached={bool(child.attached_to)}, overlay={child.is_overlay})"
                )
                continue

            child_w, child_h = child._dimensions.width, child._dimensions.height
            child_margins = child.style.margin

            # calculate position along the main axis
            start_margin = child_margins[margin_start_idx]
            main_pos = current_pos_main + start_margin
            if i > 0:  # apply gap and distribution spacing after first element
                main_pos += self.layout.gap + spacing

            # calculate position along the cross axis based on align_items
            cross_pos_start = content_y if is_row else content_x
            cross_avail = content_h if is_row else content_w
            cross_margins_total = (
                child_margins[cross_margin_start_idx] + child_margins[cross_margin_end_idx]
            )
            cross_size = getattr(child._dimensions, cross_size_attr)
            align = self.layout.align_items

            if align in ("center", "middle"):
                cross_pos = (
                    cross_pos_start
                    + child_margins[cross_margin_start_idx]
                    + (cross_avail - cross_margins_total - cross_size) / 2.0
                )
            elif align in ("end", "bottom", "right"):  # align 'end' varies by direction
                cross_pos = (
                    cross_pos_start + cross_avail - child_margins[cross_margin_end_idx] - cross_size
                )
            else:  # 'start', 'stretch', 'top', 'left'
                cross_pos = cross_pos_start + child_margins[cross_margin_start_idx]

            # set the calculated origin (relative to parent content area)
            child_origin_x = main_pos if is_row else cross_pos
            child_origin_y = cross_pos if is_row else main_pos
            child._layout_origin_in_parent = (child_origin_x, child_origin_y)
            self._log_debug(
                f"  positioned '{child.id or type(child).__name__}' layout origin: ({child_origin_x:.1f}, {child_origin_y:.1f})"
            )

            # advance main axis position for the next child
            current_pos_main = (
                main_pos + getattr(child._dimensions, size_attr) + child_margins[margin_end_idx]
            )

    def render(self, renderer: "BaseRenderer", context: Any, matrix: np.ndarray):
        """render container background/border, then children (non-overlays first)."""
        if not self.show:
            return

        # render container background/border
        if self.style.background_color or (self.style.border_color and self.style.border_width > 0):
            renderer.render_rectangle(context, self._dimensions, self.style, matrix, component=self)

        if self.debug:
            renderer.render_debug(context, self, matrix)  # container's own debug box

        # render non-overlay children relative to this container's world matrix
        non_overlays = self._layout_children_cache
        if non_overlays:
            # self._log_debug(f"rendering {len(non_overlays)} non-overlay children")
            for child in non_overlays:
                if child and child.show:
                    # child calculates its world matrix based on parent's world matrix
                    child_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
                    child.render(renderer, context, child_matrix)

        # render overlay children relative to this container's world matrix
        overlays = self._overlay_children_cache
        if overlays:
            # self._log_debug(f"rendering {len(overlays)} overlay children")
            for child in overlays:
                if child and child.show:
                    child_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
                    child.render(renderer, context, child_matrix)


# --- Explicitly rebuild models to resolve forward references ---
# This ensures Pydantic updates type hints like Component.parent referencing Container
from .component import Component, Overlay  # Ensure necessary imports

Component.model_rebuild(force=True)
Container.model_rebuild(force=True)
Overlay.model_rebuild(force=True)  # Rebuild Overlay too as it inherits from Component
# --- End model rebuilding ---
