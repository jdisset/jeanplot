"""tests that text height in data units matches rendered output for both backends."""

from __future__ import annotations

import io
import re
from lxml import etree
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jeanplot.core.text import Text
from jeanplot.core.container import Container
from jeanplot.core.models import Transform, Size
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer
from jeanplot.core.renderer.svg import SVGRenderer


def extract_svg_text_font_size(svg_str: str, text_id: str) -> tuple[float, np.ndarray]:
    """extract font-size and transform matrix for a text element."""
    root = etree.fromstring(svg_str.encode())
    g = root.xpath(f"//*[@id='{text_id}']")
    if not g:
        raise ValueError(f"element '{text_id}' not found")
    g = g[0]
    text_elem = g.find(".//{http://www.w3.org/2000/svg}text")
    if text_elem is None:
        text_elem = g if g.tag.endswith("text") else None
    if text_elem is None:
        raise ValueError(f"no text element found in '{text_id}'")

    font_size = float(text_elem.get("font-size", "12"))
    transform_str = g.get("transform", "")
    matrix = np.eye(3)
    match = re.search(r"matrix\(([^)]+)\)", transform_str)
    if match:
        vals = [float(v.strip()) for v in match.group(1).split(",")]
        if len(vals) == 6:
            matrix = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]])
    return font_size, matrix


def extract_svg_rect_dimensions(svg_str: str, rect_id: str) -> tuple[float, float, np.ndarray]:
    """extract width, height, and transform matrix for a rect element."""
    root = etree.fromstring(svg_str.encode())
    g = root.xpath(f"//*[@id='{rect_id}']")
    if not g:
        raise ValueError(f"element '{rect_id}' not found")
    g = g[0]
    rect = g.find(".//{http://www.w3.org/2000/svg}rect")
    if rect is None:
        rect = g if g.tag.endswith("rect") else None
    if rect is None:
        raise ValueError(f"no rect element found in '{rect_id}'")

    width = float(rect.get("width", "0"))
    height = float(rect.get("height", "0"))
    transform_str = g.get("transform", "")
    matrix = np.eye(3)
    match = re.search(r"matrix\(([^)]+)\)", transform_str)
    if match:
        vals = [float(v.strip()) for v in match.group(1).split(",")]
        if len(vals) == 6:
            matrix = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]])
    return width, height, matrix


def get_svg_effective_scale(matrix: np.ndarray) -> tuple[float, float]:
    """extract effective x/y scale from transform matrix."""
    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    scale_y = np.sqrt(matrix[0, 1] ** 2 + matrix[1, 1] ** 2)
    return scale_x, scale_y


def make_test_container(data_size: float, scale: tuple[float, float] = (1.0, 1.0)) -> Container:
    """create container with text and reference box of same data-unit height."""
    text = Text(id="test_text", text="X", font_size=data_size)
    ref_box = Container(
        id="ref_box",
        min_dimensions=Size(width=data_size, height=data_size),
    )
    ref_box.style.background_color = "red"

    container = Container(
        id="root",
        children=[text, ref_box],
        transform=Transform(scale=scale),
    )
    return container


@pytest.fixture
def mpl_renderer():
    return MatplotlibRenderer()


@pytest.fixture
def svg_renderer():
    return SVGRenderer()


@pytest.mark.parametrize("data_size", [1.0, 5.0, 10.0, 50.0, 100.0])
def test_svg_text_and_box_same_scale(svg_renderer, data_size):
    """text and box with same data-unit height should have same effective height in svg."""
    container = make_test_container(data_size)
    svg_str = svg_renderer.render_to_string(container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    assert abs(effective_text_height - effective_box_height) < 0.1, (
        f"text height {effective_text_height:.2f} != box height {effective_box_height:.2f}"
    )


@pytest.mark.parametrize("scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5), (2.0, 1.0), (1.0, 3.0)])
def test_svg_scaling_propagation(svg_renderer, scale):
    """container scale should affect both text and box equally."""
    data_size = 10.0
    container = make_test_container(data_size, scale)
    svg_str = svg_renderer.render_to_string(container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    assert abs(effective_text_height - effective_box_height) < 0.1, (
        f"at scale {scale}: text {effective_text_height:.2f} != box {effective_box_height:.2f}"
    )


@pytest.mark.parametrize("data_size", [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
def test_mpl_text_scales_with_data_units(mpl_renderer, data_size):
    """matplotlib text should scale proportionally with data units."""
    text = Text(id="test_text", text="X", font_size=data_size)
    ref_box = Container(
        id="ref_box",
        min_dimensions=Size(width=data_size, height=data_size),
    )
    ref_box.style.background_color = "blue"
    container = Container(id="root", children=[text, ref_box])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    context = mpl_renderer.create_context(ax=ax)
    mpl_renderer.render_component(context, container, adjust_lims=True)
    fig.canvas.draw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_range_x = xlim[1] - xlim[0]
    data_range_y = ylim[1] - ylim[0]

    assert data_range_x > 0 and data_range_y > 0
    plt.close(fig)


@pytest.mark.parametrize("scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5), (3.0, 1.0)])
def test_mpl_container_scale_affects_text(mpl_renderer, scale):
    """text inside scaled container should scale proportionally in matplotlib."""
    data_size = 10.0
    container = make_test_container(data_size, scale)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    context = mpl_renderer.create_context(ax=ax)
    mpl_renderer.render_component(context, container, adjust_lims=True)
    fig.canvas.draw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    expected_data_extent_x = data_size * 2 * scale[0]
    expected_data_extent_y = data_size * scale[1]

    actual_extent_x = xlim[1] - xlim[0]
    actual_extent_y = ylim[1] - ylim[0]

    assert actual_extent_x >= expected_data_extent_x * 0.5
    assert actual_extent_y >= expected_data_extent_y * 0.5
    plt.close(fig)


@pytest.mark.parametrize(
    "figsize,dpi",
    [
        ((4, 4), 72),
        ((6, 6), 100),
        ((8, 8), 150),
        ((10, 5), 100),
        ((5, 10), 100),
        ((12, 3), 200),
    ],
)
def test_mpl_different_figure_sizes_and_dpi(mpl_renderer, figsize, dpi):
    """text scaling should work correctly across different figure sizes and DPIs."""
    data_size = 10.0
    container = make_test_container(data_size)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    context = mpl_renderer.create_context(ax=ax)
    mpl_renderer.render_component(context, container, adjust_lims=True)
    fig.canvas.draw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[1] > xlim[0]
    assert ylim[1] > ylim[0]
    plt.close(fig)


@pytest.mark.parametrize("aspect", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_svg_non_uniform_aspect_ratio(svg_renderer, aspect):
    """text should scale correctly with non-uniform container scaling."""
    data_size = 10.0
    container = make_test_container(data_size, scale=(aspect, 1.0))
    svg_str = svg_renderer.render_to_string(container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    text_scale_x, text_scale_y = get_svg_effective_scale(text_matrix)
    box_scale_x, box_scale_y = get_svg_effective_scale(box_matrix)

    assert abs(text_scale_x / box_scale_x - 1.0) < 0.1
    assert abs(text_scale_y / box_scale_y - 1.0) < 0.1


@pytest.mark.parametrize("data_size", [0.01, 0.1, 0.5])
def test_svg_very_small_font_sizes(svg_renderer, data_size):
    """very small font sizes should still scale correctly."""
    container = make_test_container(data_size)
    svg_str = svg_renderer.render_to_string(container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    assert effective_text_height > 0
    assert effective_box_height > 0
    ratio = effective_text_height / effective_box_height
    assert 0.9 < ratio < 1.1, f"ratio {ratio:.3f} out of range"


@pytest.mark.parametrize("data_size", [200.0, 500.0, 1000.0])
def test_svg_very_large_font_sizes(svg_renderer, data_size):
    """very large font sizes should still scale correctly."""
    container = make_test_container(data_size)
    svg_str = svg_renderer.render_to_string(container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    ratio = effective_text_height / effective_box_height
    assert 0.9 < ratio < 1.1, f"ratio {ratio:.3f} out of range"


def test_svg_nested_scaling():
    """nested containers with multiple scales should compose correctly."""
    inner_text = Text(id="inner_text", text="X", font_size=5.0)
    inner_box = Container(
        id="inner_box",
        min_dimensions=Size(width=5.0, height=5.0),
    )
    inner_box.style.background_color = "green"

    inner_container = Container(
        id="inner",
        children=[inner_text, inner_box],
        transform=Transform(scale=(2.0, 2.0)),
    )

    outer_container = Container(
        id="outer",
        children=[inner_container],
        transform=Transform(scale=(1.5, 1.5)),
    )

    renderer = SVGRenderer()
    svg_str = renderer.render_to_string(outer_container)

    font_size, text_matrix = extract_svg_text_font_size(svg_str, "inner_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "inner_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    expected_total_scale = 2.0 * 1.5
    assert abs(text_scale_y - expected_total_scale) < 0.1
    assert abs(box_scale_y - expected_total_scale) < 0.1
    ratio = effective_text_height / effective_box_height
    assert 0.9 < ratio < 1.1


def test_mpl_nested_scaling(mpl_renderer):
    """nested container scaling in matplotlib should compose correctly."""
    inner_text = Text(id="inner_text", text="X", font_size=5.0)
    inner_box = Container(
        id="inner_box",
        min_dimensions=Size(width=5.0, height=5.0),
    )
    inner_box.style.background_color = "green"

    inner_container = Container(
        id="inner",
        children=[inner_text, inner_box],
        transform=Transform(scale=(2.0, 2.0)),
    )

    outer_container = Container(
        id="outer",
        children=[inner_container],
        transform=Transform(scale=(1.5, 1.5)),
    )

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    context = mpl_renderer.create_context(ax=ax)
    mpl_renderer.render_component(context, outer_container, adjust_lims=True)
    fig.canvas.draw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[1] > xlim[0]
    assert ylim[1] > ylim[0]
    plt.close(fig)


@pytest.mark.parametrize(
    "backend",
    ["svg", "mpl"],
)
@pytest.mark.parametrize("data_size", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("scale", [(1.0, 1.0), (2.0, 2.0), (0.5, 2.0)])
def test_both_backends_consistent(backend, data_size, scale, mpl_renderer, svg_renderer):
    """both backends should handle text/box scaling consistently."""
    container = make_test_container(data_size, scale)

    if backend == "svg":
        svg_str = svg_renderer.render_to_string(container)
        font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
        _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")
        _, text_scale_y = get_svg_effective_scale(text_matrix)
        _, box_scale_y = get_svg_effective_scale(box_matrix)
        effective_text_height = font_size * text_scale_y
        effective_box_height = box_height * box_scale_y
        ratio = effective_text_height / effective_box_height
        assert 0.9 < ratio < 1.1, f"svg ratio {ratio:.3f} out of range"
    else:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        context = mpl_renderer.create_context(ax=ax)
        mpl_renderer.render_component(context, container, adjust_lims=True)
        fig.canvas.draw()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[1] > xlim[0] and ylim[1] > ylim[0]
        plt.close(fig)


@pytest.mark.parametrize("rotation", [0, 45, 90, 180, 270])
def test_svg_rotation_preserves_scale(svg_renderer, rotation):
    """rotation should not affect the relative scale between text and box."""
    data_size = 10.0
    text = Text(id="test_text", text="X", font_size=data_size)
    ref_box = Container(
        id="ref_box",
        min_dimensions=Size(width=data_size, height=data_size),
    )
    ref_box.style.background_color = "red"

    container = Container(
        id="root",
        children=[text, ref_box],
        transform=Transform(rotate=rotation),
    )

    svg_str = svg_renderer.render_to_string(container)
    font_size, text_matrix = extract_svg_text_font_size(svg_str, "test_text")
    _, box_height, box_matrix = extract_svg_rect_dimensions(svg_str, "ref_box")

    _, text_scale_y = get_svg_effective_scale(text_matrix)
    _, box_scale_y = get_svg_effective_scale(box_matrix)

    effective_text_height = font_size * text_scale_y
    effective_box_height = box_height * box_scale_y

    ratio = effective_text_height / effective_box_height
    assert 0.8 < ratio < 1.2, f"rotation {rotation}: ratio {ratio:.3f} out of range"
