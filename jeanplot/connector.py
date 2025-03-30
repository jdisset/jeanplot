from typing import Optional, Tuple, Literal, Union, Dict, Any
from pydantic import Field, PrivateAttr
import numpy as np

from .component import Component
from .models import Size, Offset, Transform
from .svg import (
    SVGElement,
    make_svg_bezier,
    LineEndArrow,
    LineEndType,
    LineStyle,
)

CurveType = Literal["straight", "bezier", "orthogonal"]
# --
# straight: well, a straight line
# bezier: a smooth curve with control points
# orthogonal: a right-angle curve with control points

class CurveDefinition(BaseModel):
    # TODO
    pass



class Connection(Component):
    """connects two components with a styled line/curve"""

    start_component: Component
    end_component: Component

    start_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.0, 0.0)))
    end_offset: Offset = Field(default_factory=lambda: Offset(relative=(0.0, 0.0)))

    color: str = "#000000"
    width: float = 1.0  # aka thickness

    curve_type: CurveType = "straight"
    line_style: LineStyle = "solid"
    dash_array: Optional[Tuple[float, ...]] = None
    dash_offset: float = 0.0

    control_distance: float = 50.0
    curvature: float = 1.0

    start_cap: Optional[LineEndType] = None
    end_cap: Optional[LineEndType] = None

    is_overlay: bool = True

    _svg_element: Optional[SVGElement] = PrivateAttr(default=None)

    # TODO...
