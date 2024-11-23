from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.path import Path
from .models import (
    ContainerStyle,
    Bounds,
    Transform,
    AlignType,
    FlexDirectionType,
    PositionType,
    VerticalAlignType,
)
from pydantic import BaseModel, Field


class Component(BaseModel):
    """base component class with transformation and rendering capabilities"""

    id: str
    position: PositionType = "relative"
    transform: Transform = Field(default_factory=Transform)
    style: ContainerStyle = Field(default_factory=ContainerStyle)
    bounds: Bounds = Field(default_factory=Bounds)
    debug: bool = False

    def get_transform_matrix(self, parent_transform: Optional[np.ndarray] = None) -> np.ndarray:
        local_transform = self.transform.to_matrix()
        if parent_transform is not None:
            return parent_transform @ local_transform
        return local_transform

    def render_debug(self, ax: plt.Axes, transform: np.ndarray):
        if self.debug:
            rect = mpatches.Rectangle(
                (self.bounds.x, self.bounds.y),
                self.bounds.width,
                self.bounds.height,
                fill=False,
                edgecolor="red",
                linestyle="--",
                transform=mtransforms.Affine2D(matrix=transform) + ax.transData,
            )
            ax.add_patch(rect)
            ax.text(
                self.bounds.x,
                self.bounds.y + self.bounds.height,
                f"{self.id}\n({self.bounds.x:.1f}, {self.bounds.y:.1f})",
                color="red",
                fontsize=8,
                transform=mtransforms.Affine2D(matrix=transform) + ax.transData,
            )

    def render(self, ax: plt.Axes, parent_transform: Optional[np.ndarray] = None):
        transform = self.get_transform_matrix(parent_transform)
        self.render_debug(ax, transform)


class Text(Component):
    """renders text with alignment and styling options"""

    text: str
    font_size: float = 12
    color: str = "black"
    align: AlignType = "left"
    vertical_align: VerticalAlignType = "center"

    def render(self, ax: plt.Axes, parent_transform: Optional[np.ndarray] = None):
        transform = self.get_transform_matrix(parent_transform)

        # horizontal alignment remains the same
        x = self.bounds.x
        if self.align == "center":
            x += self.bounds.width / 2
        elif self.align == "right":
            x += self.bounds.width

        # adjust vertical position based on alignment
        y = self.bounds.y
        if self.vertical_align == "top":
            # move down from top by approximately one line height
            y += self.font_size * 0.8
        elif self.vertical_align == "center":
            y += self.bounds.height / 2
        elif self.vertical_align == "bottom":
            y += self.bounds.height - self.font_size * 0.2

        ax.text(
            x,
            y,
            self.text,
            fontsize=self.font_size,
            color=self.color,
            horizontalalignment=self.align,
            verticalalignment="baseline",  # use baseline alignment for consistent rendering
            transform=mtransforms.Affine2D(matrix=transform) + ax.transData,
        )


class Container(Component):
    direction: FlexDirectionType = "row"
    children: List[Component] = Field(default_factory=list)
    spacing: float = 0.0

    def render(self, ax: plt.Axes, parent_transform: Optional[np.ndarray] = None):
        transform = self.get_transform_matrix(parent_transform)
        self.layout()
        self.calculate_bounds()

        if self.style.background_color or self.style.border_color:
            ls = (
                (0, self.style.dash_sequence)
                if self.style.border_style == "custom" and self.style.dash_sequence
                else self.style.get_matplotlib_linestyle()
            )
            box = mpatches.PathPatch(
                self._create_rounded_rectangle_path(),
                facecolor=self.style.background_color or "none",
                edgecolor=self.style.border_color or "none",
                linewidth=self.style.border_width,
                linestyle=ls,
                transform=mtransforms.Affine2D(matrix=transform) + ax.transData,
            )
            ax.add_patch(box)

        self.render_debug(ax, transform)
        for child in self.children:
            child.render(ax, transform)

    def calculate_bounds(self):
        if not self.children:
            return

        coords = [
            (c.bounds.x, c.bounds.y, c.bounds.x + c.bounds.width, c.bounds.y + c.bounds.height)
            for c in self.children
        ]
        min_x = min(c[0] for c in coords) - self.style.padding[3]
        min_y = min(c[1] for c in coords) - self.style.padding[0]
        max_x = max(c[2] for c in coords) + self.style.padding[1]
        max_y = max(c[3] for c in coords) + self.style.padding[2]

        if len(self.children) > 1:
            if self.direction == "row":
                max_x += self.spacing * (len(self.children) - 1)
            else:
                max_y += self.spacing * (len(self.children) - 1)

        if self.position == "relative":
            self.bounds.x = min_x
            self.bounds.y = min_y
        self.bounds.width = max_x - min_x
        self.bounds.height = max_y - min_y

    def _create_rounded_rectangle_path(self) -> Path:
        x, y = self.bounds.x, self.bounds.y
        w, h = self.bounds.width, self.bounds.height
        r = min(self.style.corner_radius, w / 2, h / 2)

        if r == 0:
            return Path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])

        c = 0.552284749831
        verts = [
            (x + r, y),
            (x + w - r, y),
            (x + w - r + c * r, y),
            (x + w, y + r - c * r),
            (x + w, y + r),
            (x + w, y + h - r),
            (x + w, y + h - r + c * r),
            (x + w - r + c * r, y + h),
            (x + w - r, y + h),
            (x + r, y + h),
            (x + r - c * r, y + h),
            (x, y + h - r + c * r),
            (x, y + h - r),
            (x, y + r),
            (x, y + r - c * r),
            (x + r - c * r, y),
            (x + r, y),
        ]
        codes = [Path.MOVETO] + [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4] * 4
        return Path(verts, codes)

    def layout(self):
        if not self.children:
            return

        pos = [self.bounds.x + self.style.padding[3], self.bounds.y + self.style.padding[0]]

        for i, child in enumerate(self.children):
            if child.position == "relative":
                child.bounds.x = pos[0]
                child.bounds.y = pos[1]

                if i < len(self.children) - 1:
                    if self.direction == "row":
                        pos[0] += child.bounds.width + max(0.1, self.spacing)
                    else:
                        pos[1] += child.bounds.height + max(0.1, self.spacing)


class Card(Container):
    title: str
    title_align: AlignType = "center"

    def __init__(self, **data):
        super().__init__(**data)

        style_data = data.get("style")
        if style_data:
            default_style = ContainerStyle(
                background_color="white",
                border_color="black",
                border_width=1.0,
                padding=(10, 10, 10, 10),
                border_style="solid",
            )
            self.style = default_style.copy(update=style_data.dict())

        title_width = self.bounds.width - (self.style.padding[1] + self.style.padding[3])
        self.children.insert(
            0,
            Text(
                id=f"{self.id}_title",
                text=self.title,
                font_size=14,
                align=self.title_align,
                bounds=Bounds(
                    x=self.style.padding[3], y=self.style.padding[0], width=title_width, height=20
                ),
            ),
        )
