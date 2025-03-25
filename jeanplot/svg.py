from typing import Optional, Union, Any, List, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import numpy as np
import re
from lxml import etree
from .component import Component
from .models import Size


class SVGPathData(BaseModel):
    """Represents a single SVG path with its attributes"""

    d: str
    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = 1.0
    transform: Optional[str] = None
    is_main_color: bool = False
    is_secondary_color: bool = False


class SVGContent(BaseModel):
    """Structured representation of SVG data"""

    width: float = 100
    height: float = 100
    viewBox: Optional[Tuple[float, float, float, float]] = None
    paths: List[SVGPathData] = Field(default_factory=list)


def get_svg_data_from_string(
    svg_content: str,
    ppi: float = 1.0,
    main_color: str = "#0000FF",
    secondary_color: str = "#00FF00",
) -> SVGContent:
    """Extract SVG data from a string containing SVG content"""
    try:
        root = etree.fromstring(svg_content.encode("utf-8"))

        width_str = root.attrib.get("width", "100")
        height_str = root.attrib.get("height", "100")

        # remove 'px' suffix if present
        width = float(re.match(r"[\d\.]+", width_str).group()) / ppi
        height = float(re.match(r"[\d\.]+", height_str).group()) / ppi

        # extract viewBox
        viewBox = None
        if "viewBox" in root.attrib:
            try:
                viewBox = tuple(float(x) for x in root.attrib["viewBox"].split())
            except ValueError:
                pass

        # extract paths
        paths = []
        for elem in root.findall(".//{http://www.w3.org/2000/svg}path"):
            try:
                fill = elem.attrib.get("fill", "none")
                stroke = elem.attrib.get("stroke", "none")
                stroke_width = float(elem.attrib.get("stroke-width", 1.0))
                transform = elem.attrib.get("transform")
                is_main_color = fill == main_color
                is_secondary_color = fill == secondary_color

                path_data = SVGPathData(
                    d=elem.attrib["d"],
                    fill=fill,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    transform=transform,
                    is_main_color=is_main_color,
                    is_secondary_color=is_secondary_color,
                )
                paths.append(path_data)
            except Exception:
                continue

        return SVGContent(
            width=width,
            height=height,
            viewBox=viewBox,
            paths=paths,
        )
    except Exception as e:
        print(f"Failed to parse SVG string: {e}")
        return SVGContent()


def get_svg_data_from_file(
    file_path: Union[str, Path],
    ppi: float = 1.0,
    main_color: str = "#0000FF",
    secondary_color: str = "#00FF00",
) -> SVGContent:
    """Extract SVG data from a file"""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SVG file not found: {path.absolute()}")

        with open(path, "r") as f:
            svg_content = f.read()

        return get_svg_data_from_string(svg_content, ppi, main_color, secondary_color)
    except Exception as e:
        print(f"Failed to load SVG file: {e}")
        return SVGContent()


class SVGElement(Component):
    """svg element loaded from a file"""

    main_color: str = "black"
    secondary_color: str = "gray"
    svg_content: Union[str, Path, SVGContent] = Field(default_factory=SVGContent)

    @model_validator(mode="after")
    def load_svg_data(self):
        """load svg file and extract paths and viewBox"""
        if isinstance(self.svg_content, (str, Path)):
            try:
                self.svg_content = get_svg_data_from_file(self.svg_content)
            except Exception as e:
                print(f"Error loading SVG: {e}")
                self.svg_content = SVGContent()

        assert isinstance(self.svg_content, SVGContent), "Invalid SVG data"

        self._dimensions = Size(
            width=self.svg_content.width,
            height=self.svg_content.height,
        )
        self._transformed_aabb = self.compute_transformed_aabb()
        return self

    def measure(self, renderer=None) -> Size:
        """return the natural size of the svg"""
        self._transformed_aabb = self.compute_transformed_aabb()
        return self._dimensions

    def render(self, renderer, context, matrix: np.ndarray):
        """render svg using the provided renderer"""
        renderer.render_svg(context, self, matrix)

        if self.debug:
            renderer.render_debug(context, self, matrix)
