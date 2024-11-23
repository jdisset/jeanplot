from typing import Optional, Union, Tuple, Set, List
from pathlib import Path as FilePath
from pydantic import BaseModel, Field
from lxml import etree
from io import StringIO
import re
import numpy as np
import matplotlib.collections as mcollections
import matplotlib.transforms as mtransforms
from svgpath2mpl import parse_path
import matplotlib.pyplot as plt
from .components import Component


class SVGPath(BaseModel):
    path_data: str
    fill: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: float = 1.0
    transform: Optional[str] = None

    def get_mpl_path(self, viewBox: Optional[Tuple[float, float, float, float]] = None):
        path = parse_path(self.path_data)
        if viewBox is not None:
            vx, vy, vw, vh = viewBox
            path.vertices[:, 0] -= vx
            path.vertices[:, 1] -= vy
        return path


class SVGDefinition(BaseModel):
    """handles loading and parsing of SVG files with support for color indexing"""

    width: float
    height: float
    viewBox: Optional[Tuple[float, float, float, float]] = None
    paths: List[SVGPath] = Field(default_factory=list)
    main_color_indices: Set[int] = Field(default_factory=set)
    secondary_color_indices: Set[int] = Field(default_factory=set)

    @classmethod
    def from_file(cls, file_path: Union[str, FilePath], ppi: float = 1.0):
        path = FilePath(file_path)
        if not path.exists():
            raise FileNotFoundError(f"svg file not found: {path.absolute()}")

        try:
            tree = etree.parse(StringIO(path.read_text()))
        except Exception as e:
            raise ValueError(f"failed to parse svg file {file_path}: {str(e)}")

        root = tree.getroot()
        width = float(re.match(r"\d+", root.attrib["width"]).group()) / ppi
        height = float(re.match(r"\d+", root.attrib["height"]).group()) / ppi

        viewBox = None
        if "viewBox" in root.attrib:
            try:
                viewBox = tuple(float(x) for x in root.attrib["viewBox"].split())
            except ValueError:
                pass

        paths = []
        main_color_indices = set()
        secondary_color_indices = set()

        for i, elem in enumerate(root.findall(".//{http://www.w3.org/2000/svg}path")):
            try:
                path = SVGPath(
                    path_data=elem.attrib["d"],
                    fill=elem.attrib.get("fill", "none"),
                    stroke=elem.attrib.get("stroke", "none"),
                    stroke_width=float(elem.attrib.get("stroke-width", 1.0)),
                    transform=elem.attrib.get("transform"),
                )
                paths.append(path)
                if path.fill == "#0000FF":
                    main_color_indices.add(i)
                elif path.fill == "#00FF00":
                    secondary_color_indices.add(i)
            except Exception:
                continue

        return cls(
            width=width,
            height=height,
            viewBox=viewBox,
            paths=paths,
            main_color_indices=main_color_indices,
            secondary_color_indices=secondary_color_indices,
        )


class SVG(Component):
    """renders SVG paths as matplotlib collections with support for color theming"""

    definition: SVGDefinition
    main_color: str = "black"
    secondary_color: str = "gray"
    edge_color: Optional[str] = None
    line_width: Optional[float] = None
    preserve_aspect_ratio: bool = True

    @classmethod
    def from_file(cls, file_path: Union[str, FilePath], id: str, **kwargs):
        definition = SVGDefinition.from_file(file_path)
        return cls(id=id, definition=definition, **kwargs)

    def _get_collection(self, transform: np.ndarray, ax: plt.Axes) -> mcollections.PathCollection:
        paths = []
        for svg_path in self.definition.paths:
            path = svg_path.get_mpl_path(self.definition.viewBox)
            path.vertices[:, 1] = self.definition.height - path.vertices[:, 1]
            paths.append(path)

        facecolors = []
        for i, path in enumerate(self.definition.paths):
            if i in self.definition.main_color_indices:
                color = self.main_color
            elif i in self.definition.secondary_color_indices:
                color = self.secondary_color
            else:
                color = path.fill if path.fill != "none" else None
            facecolors.append(color)

        edgecolors = [
            self.edge_color if self.edge_color else path.stroke for path in self.definition.paths
        ]

        return mcollections.PathCollection(
            paths,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=self.line_width if self.line_width else 1.0,
            capstyle="round",
            transform=mtransforms.Affine2D(matrix=transform) + ax.transData,
        )

    def render(self, ax: plt.Axes, parent_transform: Optional[np.ndarray] = None):
        transform = self.get_transform_matrix(parent_transform)

        if self.definition.viewBox:
            vx, vy, vw, vh = self.definition.viewBox
            scale_x = self.bounds.width / vw
            scale_y = self.bounds.height / vh

            if self.preserve_aspect_ratio:
                scale = min(scale_x, scale_y)
                scale_x = scale_y = scale

            transform = transform @ np.array(
                [
                    [scale_x, 0, self.bounds.x + vx * scale_x],
                    [0, scale_y, self.bounds.y + vy * scale_y],
                    [0, 0, 1],
                ]
            )

        ax.add_collection(self._get_collection(transform, ax))
