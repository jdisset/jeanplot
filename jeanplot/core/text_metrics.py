"""Shared text measurement helpers."""

from __future__ import annotations

from functools import lru_cache

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text as MplText

from jeanplot.core.models import TextMetrics

DEFAULT_TEXT_REF_FONT_SIZE = 10.0
DEFAULT_TEXT_MEASURE_DPI = 100


def build_font_properties(
    font_name: str | None = None,
    font_weight: str = "normal",
    font_style: str = "normal",
) -> FontProperties:
    family = [font_name] if font_name else ["sans-serif"]
    return FontProperties(family=family, weight=font_weight, style=font_style)


@lru_cache(maxsize=4096)
def _measure_points_cached(
    text: str,
    font_name: str,
    font_weight: str,
    font_style: str,
    ref_font_size: float,
    line_spacing: float,
    dpi: int,
) -> tuple[float, float]:
    fig = Figure(dpi=dpi)
    renderer = RendererAgg(1, 1, dpi)
    mpl_text = MplText(
        x=0.0,
        y=0.0,
        text=text,
        fontsize=ref_font_size,
        fontproperties=build_font_properties(font_name or None, font_weight, font_style),
        linespacing=max(0.0, 1.0 + line_spacing),
    )
    mpl_text.set_figure(fig)
    bbox = mpl_text.get_window_extent(renderer=renderer)
    points_per_pixel = 72.0 / dpi
    return bbox.width * points_per_pixel, bbox.height * points_per_pixel


def measure_text_metrics(
    text: str,
    font_name: str | None = None,
    font_weight: str = "normal",
    font_style: str = "normal",
    line_spacing: float = 0.2,
    ref_font_size: float = DEFAULT_TEXT_REF_FONT_SIZE,
    dpi: int = DEFAULT_TEXT_MEASURE_DPI,
) -> TextMetrics:
    if not text:
        return TextMetrics(
            ref_font_size=ref_font_size,
            width_points=0.0,
            height_points=0.0,
        )

    width_points, height_points = _measure_points_cached(
        text=text,
        font_name=font_name or "",
        font_weight=font_weight,
        font_style=font_style,
        ref_font_size=ref_font_size,
        line_spacing=line_spacing,
        dpi=dpi,
    )
    return TextMetrics(
        ref_font_size=ref_font_size,
        width_points=width_points,
        height_points=height_points,
    )
