from pydantic import Field, PrivateAttr, model_validator
from typing import Any, Sequence
import numpy as np
import logging

from jeanplot.core.component import Component, AnchorComponent
from jeanplot.core.models import Size, LayoutConstraints, LayoutConstraintsField
from jeanplot.core.renderer import BaseRenderer

logger = logging.getLogger(__name__)


class Container(Component):
    """component that holds, lays out, and renders child components."""

    children: list[Component] = Field(default_factory=list)
    layout: LayoutConstraintsField = Field(default_factory=LayoutConstraints)

    def __init__(self, *args, **kwargs):
        if args:
            if "children" in kwargs:
                raise TypeError(
                    f"{type(self).__name__}: positional children and `children=` are mutually exclusive"
                )
            kwargs["children"] = list(args)
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _bare_list_is_children(cls, v: Any) -> Any:
        # Allow `!Container [a, b]` (or any subclass) as sugar for
        # `!Container { children: [a, b] }`.
        return {"children": v} if isinstance(v, list) else v

    _layout_children_cache: list[Component] = PrivateAttr(default_factory=list)
    _overlay_children_cache: list[Component] = PrivateAttr(default_factory=list)

    def add_child(self, child: Component):
        is_already_present = any(child is c for c in self.children)
        if not is_already_present:
            self.children.append(child)
        # Guard: `parent` is a validated field, so reassigning it (even to the same
        # value) re-runs every model_validator on the child. For a Table that means
        # build_table re-fires and wipes its computed column widths. Only assign when
        # it actually changes.
        if child.parent is not self:
            child.parent = self
        if isinstance(child, AnchorComponent):
            is_in_anchors = any(child is a for a in self.anchor_points)
            if not is_in_anchors:
                self.anchor_points.append(child)

    def add_children(self, children: Sequence[Component]):
        for child in children:
            self.add_child(child)

    def _prepare_children(self, for_render: bool = False):
        """categorize children and ensure parent links."""
        layout_children, overlay_children = [], []

        layout_ids, overlay_ids = set(), set()

        all_children_map = {id(c): c for c in self.children if c}
        for a in self.anchor_points:
            if a:
                all_children_map.setdefault(id(a), a)

        original_children_ids = {id(c) for c in self.children if c}

        for child_obj_id, child in all_children_map.items():
            # Only assign when it actually changes: `parent` is validated, so a
            # redundant reassignment re-runs every model_validator on the child each
            # layout pass (and would re-fire Table.build_table, wiping column widths).
            if child.parent is not self:
                child.parent = self
            is_overlay_or_attached = child.is_overlay or child.attached_to

            if is_overlay_or_attached:
                if child_obj_id not in overlay_ids:
                    overlay_children.append(child)
                    overlay_ids.add(child_obj_id)
            else:
                if child_obj_id not in layout_ids:
                    layout_children.append(child)
                    layout_ids.add(child_obj_id)

            if isinstance(child, AnchorComponent) and child_obj_id not in original_children_ids:
                self.children.append(child)
                original_children_ids.add(child_obj_id)

        if not for_render:
            self._layout_children_cache = layout_children
            self._overlay_children_cache = overlay_children

        return layout_children, overlay_children

    def _measure_natural(self, renderer: BaseRenderer | None) -> Size:
        """natural size based on layout children."""
        self._prepare_children(for_render=False)

        all_measurable_children = self._layout_children_cache + self._overlay_children_cache
        for child in all_measurable_children:
            if child:
                child.measure_and_layout(renderer)

        natural_width, natural_height = 0.0, 0.0
        layout_children = self._layout_children_cache
        insets = self.safe_style.content_inset()

        if not layout_children:
            natural_width, natural_height = insets[1] + insets[3], insets[0] + insets[2]
        else:
            is_row = self.layout.direction == "row"
            main_axis_size, cross_axis_size = 0.0, 0.0
            for i, child in enumerate(layout_children):
                child_dims = child._dimensions
                margins = child.safe_style.margin
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

    def _layout_children(self, renderer: BaseRenderer | None):
        """positions layout children and triggers layout in child containers."""
        layout_children, overlay_children = self._prepare_children(for_render=False)

        content_w, content_h = self.safe_style.content_box(self._dimensions)
        content_x, content_y = self.safe_style.padding[3], self.safe_style.padding[0]

        if layout_children:
            self._calculate_and_apply_stretch(content_w, content_h, layout_children)
            self._position_layout_children(
                content_x, content_y, content_w, content_h, layout_children
            )

        all_children = layout_children + overlay_children
        for child in all_children:
            if isinstance(child, Container):
                child._layout_children(renderer)

    def _layout_overlay_containers(
        self, overlay_children: list[Component], renderer: BaseRenderer | None
    ):
        for child in overlay_children:
            if isinstance(child, Container):
                child._layout_children(renderer)

    def _calculate_and_apply_stretch(
        self, content_w: float, content_h: float, layout_children: list[Component]
    ) -> list[Component]:
        if self.layout.align_items != "stretch" or not layout_children:
            return []
        stretched_children = []
        is_row = self.layout.direction == "row"
        cross_axis, main_axis = ("height", "width") if is_row else ("width", "height")
        available_cross = content_h if is_row else content_w

        for child in layout_children:
            current_dims = child._dimensions
            min_dims, max_dims = child.min_dimensions, child.max_dimensions
            margin_t, margin_r, margin_b, margin_l = child.safe_style.margin
            margins_cross = (margin_t + margin_b) if is_row else (margin_l + margin_r)
            target_cross_size = max(0, available_cross - margins_cross)
            min_cross, max_cross = getattr(min_dims, cross_axis), getattr(max_dims, cross_axis)

            final_cross_size = getattr(current_dims, cross_axis, 0.0)

            if min_cross <= target_cross_size <= max_cross:
                final_cross_size = target_cross_size
            elif target_cross_size < min_cross:
                final_cross_size = min_cross
            elif target_cross_size > max_cross:
                final_cross_size = max_cross

            min_main, max_main = getattr(min_dims, main_axis), getattr(max_dims, main_axis)
            current_main = getattr(current_dims, main_axis, 0.0)
            final_main_size = min(max(min_main, current_main), max_main)

            needs_update = False
            if abs(getattr(current_dims, cross_axis, 0.0) - final_cross_size) > 1e-6:
                setattr(child._dimensions, cross_axis, final_cross_size)
                needs_update = True
            if abs(getattr(current_dims, main_axis, 0.0) - final_main_size) > 1e-6:
                setattr(child._dimensions, main_axis, final_main_size)
                needs_update = True

            if needs_update:
                stretched_children.append(child)
        return stretched_children

    def _distribute_flex_weights(
        self,
        content_w: float,
        content_h: float,
        layout_children: list,
        is_row: bool,
        size_attr: str,
        margin_start_idx: int,
        margin_end_idx: int,
    ):
        weights = self.layout.main_axis_weights
        n = len(layout_children)
        if not weights or len(weights) != n:
            return
        total = float(sum(weights))
        if total <= 0:
            return
        available = content_w if is_row else content_h
        total_gap = self.layout.gap * max(0, n - 1)
        margins_total = sum(
            child.safe_style.margin[margin_start_idx] + child.safe_style.margin[margin_end_idx]
            for child in layout_children
        )
        flex_space = max(0.0, available - total_gap - margins_total)
        for w, child in zip(weights, layout_children):
            new_size = flex_space * (w / total)
            setattr(child._dimensions, size_attr, new_size)

    def _position_layout_children(
        self, content_x, content_y, content_w, content_h, layout_children
    ):
        """sets _layout_origin_in_parent for layout children."""
        num_children = len(layout_children)
        if num_children == 0:
            return

        is_row = self.layout.direction == "row"
        size_attr, cross_size_attr = ("width", "height") if is_row else ("height", "width")
        margin_start_idx, margin_end_idx = (3, 1) if is_row else (0, 2)
        cross_margin_start_idx, cross_margin_end_idx = (0, 2) if is_row else (3, 1)

        self._distribute_flex_weights(
            content_w,
            content_h,
            layout_children,
            is_row,
            size_attr,
            margin_start_idx,
            margin_end_idx,
        )

        total_child_main_size_with_margins = sum(
            getattr(child._dimensions, size_attr, 0.0)
            + child.safe_style.margin[margin_start_idx]
            + child.safe_style.margin[margin_end_idx]
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
            child_margins = child.safe_style.margin

            start_margin = child_margins[margin_start_idx]
            main_pos = current_pos_main + start_margin
            if i > 0:
                main_pos += self.layout.gap + spacing

            cross_pos_start = content_y if is_row else content_x
            cross_avail = content_h if is_row else content_w
            cross_margins_total = (
                child_margins[cross_margin_start_idx] + child_margins[cross_margin_end_idx]
            )
            cross_size = getattr(child_dims, cross_size_attr, 0.0)
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
            else:
                cross_pos = cross_pos_start + child_margins[cross_margin_start_idx]

            child_origin_x = main_pos if is_row else cross_pos
            child_origin_y = cross_pos if is_row else main_pos
            child._layout_origin_in_parent = (child_origin_x, child_origin_y)

            current_pos_main = (
                main_pos + getattr(child_dims, size_attr, 0.0) + child_margins[margin_end_idx]
            )

    def measure_and_layout(self, renderer: BaseRenderer | None = None) -> Size:
        comp_id_str = self.id or self.__class__.__name__
        try:
            # apply style to self before measuring children
            Component._apply_style(self)
            self._resolve_attachment()
            self._natural_dimensions = self._measure_natural(renderer)
            self._dimensions = self._apply_constraints(self._natural_dimensions)
            self._layout_children(renderer)
        except Exception as e:
            logger.error(
                f"exception during measure_and_layout for {comp_id_str}: {e}", exc_info=True
            )
            self._dimensions = self._apply_constraints(Size())
        if not isinstance(self._dimensions, Size):
            logger.error(
                f"measure_and_layout for {comp_id_str} resulted in non-Size _dimensions: {type(self._dimensions)}. Fixing."
            )
            self._dimensions = Size()
        return self._dimensions

    def render(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        """render container background/border, then sorted children."""
        if not self.show:
            return

        style = self.safe_style
        if (
            style.background_color
            or (style.border_color and style.border_width > 0)
            or style.shadow
        ):
            renderer.render_rectangle(context, self._dimensions, style, matrix, component=self)

        if self.debug:
            renderer.render_debug(context, self, matrix)

        self._render_children(renderer, context, matrix)

    def _render_children(self, renderer: BaseRenderer, context: Any, matrix: np.ndarray):
        """render visible children and anchors in z-order."""
        all_children_map = {id(c): c for c in self.children if c}
        for a in self.anchor_points:
            if a:
                all_children_map.setdefault(id(a), a)

        visible_children = [child for child in all_children_map.values() if child and child.show]
        visible_children.sort(key=lambda c: getattr(c, "z_index", 0))

        if visible_children:
            for child in visible_children:
                if hasattr(child, "compute_world_matrix") and callable(child.compute_world_matrix):
                    child_world_matrix = child.compute_world_matrix(parent_world_matrix=matrix)
                    child.render(renderer, context, child_world_matrix)
                else:
                    logger.warning(
                        f"child component {type(child)} (id={getattr(child, 'id', '?')}) lacks compute_world_matrix, skipping render."
                    )
