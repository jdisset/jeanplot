# File: jeanplot/container.py
# -*- coding: utf-8 -*-
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
from jeanplot.renderer import BaseRenderer  # Use explicit import with BaseRenderer

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
        if isinstance(child, AnchorComponent) and child not in self.anchor_points:
            self.anchor_points.append(child)

    def add_children(self, children: Sequence[Component]):
        for child in children:
            self.add_child(child)

    def _prepare_children(self):
        """categorize children for layout/overlay and ensure parent links."""
        layout_children, overlay_children = [], []
        # combine explicitly added children and anchors
        all_children = self.children + [a for a in self.anchor_points if a not in self.children]

        for child in all_children:
            if child:
                child.parent = self  # ensure parent link
                if child.is_overlay or child.attached_to:
                    if child not in overlay_children:
                        overlay_children.append(child)
                else:
                    if child not in layout_children:
                        layout_children.append(child)
                # ensure anchors added via anchor_points are also in main list if needed
                if isinstance(child, AnchorComponent) and child not in self.children:
                    self.children.append(child)

        self._layout_children_cache = layout_children
        self._overlay_children_cache = overlay_children

    def _measure_natural(self, renderer: Optional[BaseRenderer]) -> Size:
        """calculates natural size based on layout children."""
        self._prepare_children()

        for child in self.children:  # measure all children first
            if child:
                child.measure_and_layout(renderer)

        natural_width, natural_height = 0.0, 0.0
        layout_children = self._layout_children_cache
        insets = self.style.content_inset()  # t, r, b, l

        if not layout_children:
            natural_width, natural_height = insets[1] + insets[3], insets[0] + insets[2]
        else:
            is_row = self.layout.direction == "row"
            main_axis_size, cross_axis_size = 0.0, 0.0
            for i, child in enumerate(layout_children):
                child_dims = child._dimensions
                margins = child.style.margin  # t, r, b, l
                if is_row:
                    main_axis_size += child_dims.width + margins[3] + margins[1]
                    cross_axis_size = max(
                        cross_axis_size, child_dims.height + margins[0] + margins[2]
                    )
                else:
                    main_axis_size += child_dims.height + margins[0] + margins[2]
                    cross_axis_size = max(
                        cross_axis_size, child_dims.width + margins[3] + margins[1]
                    )
                if i > 0:
                    main_axis_size += self.layout.gap

            natural_width, natural_height = (
                (main_axis_size, cross_axis_size) if is_row else (cross_axis_size, main_axis_size)
            )
            natural_width += insets[1] + insets[3]
            natural_height += insets[0] + insets[2]

        natural_width, natural_height = max(0.0, natural_width), max(0.0, natural_height)
        return Size(width=natural_width, height=natural_height)

    def _layout_children(self, renderer: Optional[BaseRenderer]):
        """positions layout children and triggers layout in child containers."""
        self._prepare_children()
        layout_children = self._layout_children_cache
        if not layout_children:
            self._layout_overlay_containers(renderer)
            return

        content_w, content_h = self.style.content_box(self._dimensions)
        content_x, content_y = self.style.padding[3], self.style.padding[0]

        stretched_children = self._calculate_and_apply_stretch(
            content_w, content_h, layout_children
        )
        self._position_layout_children(content_x, content_y, content_w, content_h, layout_children)

        # re-run internal layout for all children that are containers, including overlays
        # this ensures overlays with internal layout needs (like SourceAnnotation tag) get laid out
        all_children = self._layout_children_cache + self._overlay_children_cache
        for child in all_children:
            if isinstance(child, Container):
                child._layout_children(renderer)

    def _layout_overlay_containers(self, renderer: Optional[BaseRenderer]):
        """triggers layout pass for overlay children that are containers."""
        # this method might become redundant if the main _layout_children loop handles all containers
        for child in self._overlay_children_cache:
            if isinstance(child, Container):
                child._layout_children(renderer)

    def _calculate_and_apply_stretch(
        self, content_w: float, content_h: float, layout_children: List[Component]
    ) -> List[Component]:
        """calculates and applies stretch to children."""
        if self.layout.align_items != "stretch" or not layout_children:
            return []
        stretched_children = []
        is_row = self.layout.direction == "row"
        cross_axis, main_axis = ("height", "width") if is_row else ("width", "height")
        available_cross = content_h if is_row else content_w

        for child in layout_children:
            current_dims = child._dimensions
            min_dims, max_dims = child.min_dimensions, child.max_dimensions
            margin_t, margin_r, margin_b, margin_l = child.style.margin
            margins_cross = (margin_t + margin_b) if is_row else (margin_l + margin_r)
            target_cross_size = max(0, available_cross - margins_cross)
            min_cross, max_cross = getattr(min_dims, cross_axis), getattr(max_dims, cross_axis)

            final_cross_size = (
                current_dims.height
            )  # Default to avoid errors if cross_axis isn't height
            if cross_axis == "height":
                final_cross_size = current_dims.height
            elif cross_axis == "width":
                final_cross_size = current_dims.width

            if min_cross <= target_cross_size <= max_cross:
                final_cross_size = target_cross_size
            elif target_cross_size < min_cross:
                final_cross_size = min_cross
            elif target_cross_size > max_cross:
                final_cross_size = max_cross

            min_main, max_main = getattr(min_dims, main_axis), getattr(max_dims, main_axis)
            current_main = getattr(current_dims, main_axis)
            final_main_size = min(max(min_main, current_main), max_main)

            needs_update = False
            if abs(getattr(current_dims, cross_axis) - final_cross_size) > 1e-6:
                setattr(child._dimensions, cross_axis, final_cross_size)
                needs_update = True
            if abs(getattr(current_dims, main_axis) - final_main_size) > 1e-6:
                setattr(child._dimensions, main_axis, final_main_size)
                needs_update = True

            if needs_update:
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
        size_attr, cross_size_attr = ("width", "height") if is_row else ("height", "width")
        margin_start_idx, margin_end_idx = (3, 1) if is_row else (0, 2)
        cross_margin_start_idx, cross_margin_end_idx = (0, 2) if is_row else (3, 1)

        total_child_main_size_with_margins = sum(
            getattr(child._dimensions, size_attr)
            + child.style.margin[margin_start_idx]
            + child.style.margin[margin_end_idx]
            for child in layout_children
        )
        total_gap = self.layout.gap * max(0, num_children - 1)
        required_space = total_child_main_size_with_margins + total_gap
        available_space = content_w if is_row else content_h
        extra_space = max(0, available_space - required_space)

        current_pos_main = content_x if is_row else content_y
        justify = self.layout.justify_content
        spacing = 0.0

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

        for i, child in enumerate(layout_children):
            if child.attached_to or child.is_overlay:
                continue

            child_dims = child._dimensions
            child_margins = child.style.margin
            start_margin = child_margins[margin_start_idx]
            main_pos = current_pos_main + start_margin
            if i > 0:
                main_pos += self.layout.gap + spacing

            cross_pos_start = content_y if is_row else content_x
            cross_avail = content_h if is_row else content_w
            cross_margins_total = (
                child_margins[cross_margin_start_idx] + child_margins[cross_margin_end_idx]
            )
            cross_size = getattr(child_dims, cross_size_attr)
            align = self.layout.align_items

            if align in ("center", "middle"):
                cross_pos = (
                    cross_pos_start
                    + child_margins[cross_margin_start_idx]
                    + (cross_avail - cross_margins_total - cross_size) / 2.0
                )
            elif align in ("end", "bottom", "right"):
                cross_pos = (
                    cross_pos_start + cross_avail - child_margins[cross_margin_end_idx] - cross_size
                )
            else:  # 'start', 'stretch', 'top', 'left'
                cross_pos = cross_pos_start + child_margins[cross_margin_start_idx]

            child_origin_x = main_pos if is_row else cross_pos
            child_origin_y = cross_pos if is_row else main_pos
            child._layout_origin_in_parent = (child_origin_x, child_origin_y)

            current_pos_main = (
                main_pos + getattr(child_dims, size_attr) + child_margins[margin_end_idx]
            )

    def measure_and_layout(self, renderer: Optional[BaseRenderer] = None) -> Size:
        """coordinates the measurement and layout process, ensuring Size is returned."""
        comp_id_str = self.id or self.__class__.__name__
        try:
            self._apply_style()
            self._resolve_attachment()
            self._natural_dimensions = self._measure_natural(renderer)
            self._dimensions = self._apply_constraints(self._natural_dimensions)
            self._layout_children(renderer)
        except Exception as e:
            logger.error(
                f"exception during measure_and_layout for {comp_id_str}: {e}", exc_info=True
            )
            # return a default Size on error to prevent None propagation
            self._dimensions = self._apply_constraints(Size())  # ensure dimensions is a Size object
        # ensure _dimensions is always a Size object before returning
        if not isinstance(self._dimensions, Size):
            logger.error(
                f"measure_and_layout for {comp_id_str} resulted in non-Size _dimensions: {type(self._dimensions)}. Fixing."
            )
            self._dimensions = Size()  # fallback
        return self._dimensions

    def render(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        """render container background/border, then sorted children."""
        if not self.show:
            return

        if (
            self.style.background_color
            or (self.style.border_color and self.style.border_width > 0)
            or self.style.shadow
        ):
            renderer.render_rectangle(context, self._dimensions, self.style, matrix, component=self)
        if self.debug:
            renderer.render_debug(context, self, matrix)

        # re-prepare children just before rendering to ensure correct order
        # _prepare_children() # removed, caches should be valid from layout pass

        all_children = self._layout_children_cache + self._overlay_children_cache
        visible_children = [child for child in all_children if child and child.show]
        visible_children.sort(key=lambda c: getattr(c, "z_index", 0))

        if visible_children:
            for child in visible_children:
                # recalculate world matrix for rendering
                # parent_world_matrix is the matrix passed to this container's render call
                child_world_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
                child.render(renderer, context, child_world_matrix)


# --- Explicitly rebuild models ---
# Need Component for type hints
from .component import Component, Overlay

Component.model_rebuild(force=True)
Container.model_rebuild(force=True)
Overlay.model_rebuild(force=True)
