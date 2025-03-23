from typing import Optional, Union, Tuple, Set, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from lxml import etree
from io import StringIO
import re
from .models import Size, Offset


class SVGPath(BaseModel):
    """represents a single SVG path element"""

    path_data: str
    fill: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: float = 1.0
    transform: Optional[str] = None
    is_main_color: bool = False
    is_secondary_color: bool = False


class SVGDocument(BaseModel):
    """represents an SVG document with paths and metadata"""

    width: float
    height: float
    viewBox: Optional[Tuple[float, float, float, float]] = None
    paths: List[SVGPath] = Field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], ppi: float = 1.0):
        """load an SVG document from a file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SVG file not found: {path.absolute()}")

        try:
            tree = etree.parse(str(path))
        except Exception as e:
            raise ValueError(f"Failed to parse SVG file {file_path}: {str(e)}")

        root = tree.getroot()

        # extract width and height
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
                is_main_color = fill == "#0000FF"
                is_secondary_color = fill == "#00FF00"

                path = SVGPath(
                    path_data=elem.attrib["d"],
                    fill=fill,
                    stroke=elem.attrib.get("stroke", "none"),
                    stroke_width=float(elem.attrib.get("stroke-width", 1.0)),
                    transform=elem.attrib.get("transform"),
                    is_main_color=is_main_color,
                    is_secondary_color=is_secondary_color,
                )
                paths.append(path)
            except Exception:
                continue

        return cls(
            width=width,
            height=height,
            viewBox=viewBox,
            paths=paths,
        )


def get_svg_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """utility function to load SVG data from a file"""
    try:
        document = SVGDocument.from_file(file_path)

        # prepare path data in the format expected by renderers
        paths = []
        for path in document.paths:
            path_data = {
                "d": path.path_data,
                "fill": path.fill,
                "stroke": path.stroke,
                "stroke_width": path.stroke_width,
                "is_main_color": path.is_main_color,
                "is_secondary_color": path.is_secondary_color,
            }
            paths.append(path_data)

        return {
            "width": document.width,
            "height": document.height,
            "viewBox": document.viewBox,
            "paths": paths,
        }
    except Exception as e:
        print(f"Failed to load SVG: {e}")
        return {
            "width": 100,
            "height": 100,
            "viewBox": None,
            "paths": [],
        }
