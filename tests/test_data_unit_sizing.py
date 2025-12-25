"""Tests for data-unit sizing consistency across figure sizes, DPI, and aspect ratios.

These tests verify that text font_size and line widths specified in data units
maintain consistent proportions relative to other data-unit-sized elements,
regardless of output format settings.
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

import matplotlib.pyplot as plt

from jeanplot import (
    Container,
    Text,
    BoxStyle,
    LayoutConstraints,
    Transform,
    jstyle,
    Size,
)
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer
from jeanplot.core.renderer.svg import SVGRenderer


SVG_NS = "http://www.w3.org/2000/svg"


def extract_svg_stroke_widths(svg_str: str) -> list[tuple[str, float]]:
    root = etree.fromstring(svg_str.encode())
    results = []
    for elem in root.iter():
        sw = elem.get("stroke-width")
        elem_id = elem.get("id", elem.tag.split("}")[-1])
        if sw and sw != "none":
            results.append((elem_id, float(sw)))
    return results


def extract_svg_font_sizes(svg_str: str) -> list[tuple[str, float]]:
    root = etree.fromstring(svg_str.encode())
    results = []
    for elem in root.iter(f"{{{SVG_NS}}}text"):
        fs = elem.get("font-size")
        elem_id = elem.get("id", "text")
        if fs:
            results.append((elem_id, float(fs)))
    return results


def extract_svg_rect_sizes(svg_str: str) -> dict[str, tuple[float, float]]:
    root = etree.fromstring(svg_str.encode())
    results = {}
    for g in root.iter(f"{{{SVG_NS}}}g"):
        g_id = g.get("id")
        if not g_id:
            continue
        rect = g.find(f"{{{SVG_NS}}}rect")
        if rect is not None:
            w = float(rect.get("width", 0))
            h = float(rect.get("height", 0))
            results[g_id] = (w, h)
    return results


def get_svg_viewbox(svg_str: str) -> tuple[float, float, float, float]:
    root = etree.fromstring(svg_str.encode())
    vb = root.get("viewBox", "0 0 800 600")
    parts = [float(x) for x in vb.split()]
    return tuple(parts) if len(parts) == 4 else (0, 0, 800, 600)


def extract_transform_scale(svg_str: str, elem_id: str) -> float:
    root = etree.fromstring(svg_str.encode())
    elems = root.xpath(f"//*[@id='{elem_id}']")
    if not elems:
        return 1.0
    transform = elems[0].get("transform", "")
    m = re.search(r"matrix\(([^)]+)\)", transform)
    if m:
        vals = [float(v.strip()) for v in m.group(1).split(",")]
        if len(vals) >= 4:
            sx = np.sqrt(vals[0] ** 2 + vals[1] ** 2)
            sy = np.sqrt(vals[2] ** 2 + vals[3] ** 2)
            return (sx + sy) / 2
    return 1.0


@pytest.fixture
def mpl_renderer():
    return MatplotlibRenderer()


@pytest.fixture
def svg_renderer():
    return SVGRenderer()


class TestSVGDataUnitSizing:
    def test_box_and_text_proportions_consistent(self, svg_renderer):
        box = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(border_color="black", border_width=2.0),
            children=[Text(id="label", text="Hello", font_size=10.0)],
        )
        jstyle.apply(box)

        svg1 = svg_renderer.render_to_string(box)
        rects1 = extract_svg_rect_sizes(svg1)
        fonts1 = extract_svg_font_sizes(svg1)

        box_size1 = rects1.get("box", (100, 50))
        font_size1 = fonts1[0][1] if fonts1 else 10.0

        ratio1 = font_size1 / box_size1[1]

        box2 = Container(
            id="box",
            min_dimensions=Size(200, 100),
            style=BoxStyle(border_color="black", border_width=4.0),
            children=[Text(id="label", text="Hello", font_size=20.0)],
        )
        jstyle.apply(box2)

        svg_renderer2 = SVGRenderer()
        svg2 = svg_renderer2.render_to_string(box2)
        rects2 = extract_svg_rect_sizes(svg2)
        fonts2 = extract_svg_font_sizes(svg2)

        box_size2 = rects2.get("box", (200, 100))
        font_size2 = fonts2[0][1] if fonts2 else 20.0

        ratio2 = font_size2 / box_size2[1]

        assert abs(ratio1 - ratio2) < 0.01, (
            f"Font/box ratio should be consistent: {ratio1:.4f} vs {ratio2:.4f}"
        )

    def test_stroke_width_scales_with_box(self, svg_renderer):
        box1 = Container(
            id="box",
            min_dimensions=Size(50, 50),
            style=BoxStyle(border_color="black", border_width=1.0),
        )
        jstyle.apply(box1)
        svg1 = svg_renderer.render_to_string(box1)
        strokes1 = extract_svg_stroke_widths(svg1)
        rects1 = extract_svg_rect_sizes(svg1)

        box_w1 = rects1.get("box", (50, 50))[0]
        stroke1 = strokes1[0][1] if strokes1 else 1.0
        ratio1 = stroke1 / box_w1

        box2 = Container(
            id="box",
            min_dimensions=Size(100, 100),
            style=BoxStyle(border_color="black", border_width=2.0),
        )
        jstyle.apply(box2)
        svg_renderer2 = SVGRenderer()
        svg2 = svg_renderer2.render_to_string(box2)
        strokes2 = extract_svg_stroke_widths(svg2)
        rects2 = extract_svg_rect_sizes(svg2)

        box_w2 = rects2.get("box", (100, 100))[0]
        stroke2 = strokes2[0][1] if strokes2 else 2.0
        ratio2 = stroke2 / box_w2

        assert abs(ratio1 - ratio2) < 0.01, (
            f"Stroke/box ratio should be consistent: {ratio1:.4f} vs {ratio2:.4f}"
        )

    def test_nested_container_text_sizing(self, svg_renderer):
        inner = Container(
            id="inner",
            min_dimensions=Size(80, 40),
            children=[Text(id="inner_text", text="Inner", font_size=8.0)],
        )
        outer = Container(
            id="outer",
            min_dimensions=Size(200, 100),
            style=BoxStyle(padding=(10, 10, 10, 10)),
            children=[inner],
        )
        jstyle.apply(outer)

        svg = svg_renderer.render_to_string(outer)
        fonts = extract_svg_font_sizes(svg)
        rects = extract_svg_rect_sizes(svg)

        inner_size = rects.get("inner", (80, 40))
        font_size = fonts[0][1] if fonts else 8.0

        expected_ratio = 8.0 / 40.0
        actual_ratio = font_size / inner_size[1]

        assert abs(expected_ratio - actual_ratio) < 0.02, (
            f"Nested text/container ratio: expected {expected_ratio:.4f}, got {actual_ratio:.4f}"
        )


class TestMatplotlibDataUnitSizing:
    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_font_size_consistent_across_dpi(self, mpl_renderer, cleanup_figures):
        box = Container(
            id="box",
            min_dimensions=Size(100, 50),
            children=[Text(id="label", text="Test", font_size=10.0)],
        )
        jstyle.apply(box)

        font_sizes = []
        for dpi in [72, 150, 300]:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")

            mpl_renderer.create_context(ax=ax)
            box.measure_and_layout(mpl_renderer)

            text = box.children[0]
            matrix = text.compute_world_matrix()

            from jeanplot.core.renderer.matplotlib import _get_points_per_unit_vector

            ppu = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
            font_size_pts = text.font_size * ppu

            font_sizes.append((dpi, font_size_pts))
            plt.close(fig)

        ratios = [fs / font_sizes[0][1] for _, fs in font_sizes]
        for i, r in enumerate(ratios[1:], 1):
            assert abs(r - 1.0) < 0.05, (
                f"Font size should be consistent: DPI {font_sizes[i][0]} "
                f"ratio={r:.3f} vs DPI {font_sizes[0][0]}"
            )

    def test_font_size_consistent_across_figsize(self, mpl_renderer, cleanup_figures):
        box = Container(
            id="box",
            min_dimensions=Size(100, 50),
            children=[Text(id="label", text="Test", font_size=10.0)],
        )
        jstyle.apply(box)

        results = []
        for figsize in [(4, 3), (8, 6), (12, 9)]:
            fig, ax = plt.subplots(figsize=figsize, dpi=100)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")

            mpl_renderer.create_context(ax=ax)
            box.measure_and_layout(mpl_renderer)

            text = box.children[0]
            matrix = text.compute_world_matrix()

            from jeanplot.core.renderer.matplotlib import _get_points_per_unit_vector

            ppu = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
            font_size_pts = text.font_size * ppu

            results.append((figsize, font_size_pts))
            plt.close(fig)

        base = results[0][1]
        for figsize, fs in results[1:]:
            ratio = fs / base
            expected_ratio = figsize[0] / results[0][0][0]
            assert abs(ratio - expected_ratio) < 0.1, (
                f"Font size should scale with figsize: {figsize} "
                f"ratio={ratio:.3f}, expected={expected_ratio:.3f}"
            )

    def test_line_width_consistent_across_dpi(self, mpl_renderer, cleanup_figures):
        box = Container(
            id="box",
            min_dimensions=Size(100, 100),
            style=BoxStyle(border_color="black", border_width=2.0, border_width_mode="data"),
        )
        jstyle.apply(box)

        results = []
        for dpi in [72, 150, 300]:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 200)
            ax.set_aspect("equal")

            mpl_renderer.create_context(ax=ax)
            mpl_renderer.render_component(ax, box)

            patches = ax.patches
            if patches:
                lw = patches[0].get_linewidth()
                results.append((dpi, lw))

            plt.close(fig)

        base_lw = results[0][1]
        for dpi, lw in results[1:]:
            ratio = lw / base_lw
            assert 0.9 < ratio < 1.1, f"Line width ratio should be ~1: DPI {dpi} ratio={ratio:.3f}"

    def test_text_and_box_proportions_match(self, mpl_renderer, cleanup_figures):
        text_size = 10.0
        box_height = 50.0
        border_width = 2.0

        box = Container(
            id="box",
            min_dimensions=Size(100, box_height),
            style=BoxStyle(
                border_color="black", border_width=border_width, border_width_mode="data"
            ),
            children=[Text(id="label", text="Test", font_size=text_size)],
        )
        jstyle.apply(box)

        expected_text_box_ratio = text_size / box_height

        for dpi in [72, 150]:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")

            mpl_renderer.create_context(ax=ax)
            box.measure_and_layout(mpl_renderer)

            text = box.children[0]
            matrix = text.compute_world_matrix()

            from jeanplot.core.renderer.matplotlib import (
                _get_points_per_unit_vector,
            )

            ppu = _get_points_per_unit_vector(ax, matrix, vector=(0, 1))
            font_pts = text.font_size * ppu
            box_pts = box_height * ppu

            actual_ratio = font_pts / box_pts
            assert abs(actual_ratio - expected_text_box_ratio) < 0.01, (
                f"DPI {dpi}: text/box ratio {actual_ratio:.4f} != expected {expected_text_box_ratio:.4f}"
            )

            plt.close(fig)


class TestLayoutVariants:
    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_row_layout_sizing(self, cleanup_figures):
        container = Container(
            id="row",
            layout=LayoutConstraints(direction="row", gap=10),
            style=BoxStyle(border_color="blue", border_width=1.0),
            children=[
                Container(
                    id="box1",
                    min_dimensions=Size(50, 30),
                    style=BoxStyle(border_color="red", border_width=1.0),
                    children=[Text(id="t1", text="A", font_size=8.0)],
                ),
                Container(
                    id="box2",
                    min_dimensions=Size(50, 30),
                    style=BoxStyle(border_color="green", border_width=1.0),
                    children=[Text(id="t2", text="B", font_size=8.0)],
                ),
            ],
        )
        jstyle.apply(container)

        renderer = SVGRenderer()
        svg = renderer.render_to_string(container)

        fonts = extract_svg_font_sizes(svg)
        rects = extract_svg_rect_sizes(svg)

        box1_h = rects.get("box1", (50, 30))[1]
        box2_h = rects.get("box2", (50, 30))[1]

        font_sizes = [f[1] for f in fonts]

        ratio1 = font_sizes[0] / box1_h if font_sizes else 0
        ratio2 = font_sizes[1] / box2_h if len(font_sizes) > 1 else 0

        assert abs(ratio1 - ratio2) < 0.01, (
            f"Row layout: text/box ratios should match: {ratio1:.4f} vs {ratio2:.4f}"
        )

    def test_column_layout_sizing(self, cleanup_figures):
        container = Container(
            id="col",
            layout=LayoutConstraints(direction="column", gap=10),
            style=BoxStyle(border_color="blue", border_width=1.0),
            children=[
                Container(
                    id="box1",
                    min_dimensions=Size(60, 25),
                    style=BoxStyle(border_color="red", border_width=1.0),
                    children=[Text(id="t1", text="Row1", font_size=6.0)],
                ),
                Container(
                    id="box2",
                    min_dimensions=Size(60, 25),
                    style=BoxStyle(border_color="green", border_width=1.0),
                    children=[Text(id="t2", text="Row2", font_size=6.0)],
                ),
            ],
        )
        jstyle.apply(container)

        renderer = SVGRenderer()
        svg = renderer.render_to_string(container)

        fonts = extract_svg_font_sizes(svg)
        rects = extract_svg_rect_sizes(svg)

        box1_h = rects.get("box1", (60, 25))[1]
        box2_h = rects.get("box2", (60, 25))[1]

        font_sizes = [f[1] for f in fonts]

        if len(font_sizes) >= 2:
            ratio1 = font_sizes[0] / box1_h
            ratio2 = font_sizes[1] / box2_h

            assert abs(ratio1 - ratio2) < 0.01, (
                f"Column layout: text/box ratios should match: {ratio1:.4f} vs {ratio2:.4f}"
            )

    def test_scaled_container_preserves_proportions(self, cleanup_figures):
        inner = Container(
            id="inner",
            min_dimensions=Size(40, 20),
            style=BoxStyle(border_color="red", border_width=0.5),
            children=[Text(id="inner_text", text="Hi", font_size=5.0)],
        )
        outer = Container(
            id="outer",
            min_dimensions=Size(100, 60),
            style=BoxStyle(border_color="blue", border_width=1.0),
            transform=Transform(scale=(2.0, 2.0)),
            children=[inner],
        )
        jstyle.apply(outer)

        renderer = SVGRenderer()
        svg = renderer.render_to_string(outer)

        fonts = extract_svg_font_sizes(svg)
        rects = extract_svg_rect_sizes(svg)

        inner_h = rects.get("inner", (40, 20))[1]
        font_size = fonts[0][1] if fonts else 5.0

        expected_ratio = 5.0 / 20.0
        actual_ratio = font_size / inner_h

        assert abs(expected_ratio - actual_ratio) < 0.02, (
            f"Scaled container: text/box ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"
        )


class TestPDFOutput:
    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_pdf_text_and_box_proportions(self, cleanup_figures):
        pytest.importorskip("pypdf")
        from pypdf import PdfReader

        box = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(border_color="black", border_width=2.0),
            children=[Text(id="label", text="PDF Test", font_size=10.0)],
        )
        jstyle.apply(box)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, box)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)

            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                assert len(reader.pages) > 0
                page = reader.pages[0]
                assert page.mediabox.width > 0
                assert page.mediabox.height > 0
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    def test_pdf_consistent_across_dpi(self, cleanup_figures):
        pytest.importorskip("pypdf")
        from pypdf import PdfReader

        box = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(border_color="black", border_width=2.0),
            children=[Text(id="label", text="DPI Test", font_size=10.0)],
        )
        jstyle.apply(box)

        pdf_sizes = []

        for dpi in [72, 150, 300]:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")

            renderer = MatplotlibRenderer()
            renderer.create_context(ax=ax)
            renderer.render_component(ax, box)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                pdf_path = f.name

            try:
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close(fig)

                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    page = reader.pages[0]
                    pdf_sizes.append((dpi, page.mediabox.width, page.mediabox.height))
            finally:
                Path(pdf_path).unlink(missing_ok=True)

        base_w, base_h = pdf_sizes[0][1], pdf_sizes[0][2]
        for dpi, w, h in pdf_sizes[1:]:
            w_ratio = float(w) / float(base_w)
            h_ratio = float(h) / float(base_h)
            assert 0.9 < w_ratio < 1.1, (
                f"PDF width should be consistent across DPI: {dpi} ratio={w_ratio}"
            )
            assert 0.9 < h_ratio < 1.1, (
                f"PDF height should be consistent across DPI: {dpi} ratio={h_ratio}"
            )


class TestDataModeVsPointMode:
    @pytest.fixture
    def cleanup_figures(self):
        yield
        plt.close("all")

    def test_data_mode_scales_with_zoom(self, cleanup_figures):
        box = Container(
            id="box",
            min_dimensions=Size(100, 100),
            style=BoxStyle(border_color="black", border_width=5.0, border_width_mode="data"),
        )
        jstyle.apply(box)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, box, adjust_lims=False)

        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect("equal")
        fig.canvas.draw()

        patches = list(ax.patches)
        lw1 = patches[0].get_linewidth() if patches else 0

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        fig.canvas.draw()

        renderer.refresh_linewidths()
        lw2 = patches[0].get_linewidth() if patches else 0

        assert lw2 > lw1 * 1.8, (
            f"Data mode: line width should ~double with 2x zoom. lw1={lw1:.2f}, lw2={lw2:.2f}, ratio={lw2 / lw1:.2f}"
        )

        plt.close(fig)

    def test_point_mode_constant_across_zoom(self, cleanup_figures):
        box = Container(
            id="box",
            min_dimensions=Size(100, 100),
            style=BoxStyle(border_color="black", border_width=5.0, border_width_mode="point"),
        )
        jstyle.apply(box)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, box, adjust_lims=False)

        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect("equal")
        fig.canvas.draw()

        patches = list(ax.patches)
        lw1 = patches[0].get_linewidth() if patches else 5.0

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        fig.canvas.draw()

        renderer.refresh_linewidths()
        lw2 = patches[0].get_linewidth() if patches else 5.0

        assert abs(lw1 - lw2) < 0.5, (
            f"Point mode: line width should stay constant. lw1={lw1:.2f}, lw2={lw2:.2f}"
        )

        plt.close(fig)


class TestComplexLayouts:
    def test_mixed_content_proportions(self):
        layout = Container(
            id="layout",
            layout=LayoutConstraints(direction="column", gap=5),
            style=BoxStyle(border_color="gray", border_width=1.0),
            children=[
                Container(
                    id="header",
                    min_dimensions=Size(200, 30),
                    style=BoxStyle(background_color="#eee", border_color="black", border_width=0.5),
                    children=[Text(id="title", text="Header", font_size=8.0)],
                ),
                Container(
                    id="content",
                    layout=LayoutConstraints(direction="row", gap=10),
                    min_dimensions=Size(200, 80),
                    children=[
                        Container(
                            id="left",
                            min_dimensions=Size(60, 60),
                            style=BoxStyle(border_color="blue", border_width=1.0),
                            children=[Text(id="left_text", text="L", font_size=10.0)],
                        ),
                        Container(
                            id="right",
                            min_dimensions=Size(120, 60),
                            style=BoxStyle(border_color="green", border_width=1.0),
                            children=[Text(id="right_text", text="Right", font_size=10.0)],
                        ),
                    ],
                ),
                Container(
                    id="footer",
                    min_dimensions=Size(200, 20),
                    style=BoxStyle(background_color="#ddd", border_color="black", border_width=0.5),
                    children=[Text(id="footer_text", text="Footer", font_size=5.0)],
                ),
            ],
        )
        jstyle.apply(layout)

        renderer = SVGRenderer()
        svg = renderer.render_to_string(layout)

        fonts = extract_svg_font_sizes(svg)
        rects = extract_svg_rect_sizes(svg)

        font_map = {f[0]: f[1] for f in fonts}

        header_h = rects.get("header", (200, 30))[1]
        left_h = rects.get("left", (60, 60))[1]
        footer_h = rects.get("footer", (200, 20))[1]

        expected_header_ratio = 8.0 / 30.0
        expected_left_ratio = 10.0 / 60.0
        expected_footer_ratio = 5.0 / 20.0

        if "title" in font_map:
            actual_header_ratio = font_map["title"] / header_h
            assert abs(actual_header_ratio - expected_header_ratio) < 0.05

        if "left_text" in font_map:
            actual_left_ratio = font_map["left_text"] / left_h
            assert abs(actual_left_ratio - expected_left_ratio) < 0.05

        if "footer_text" in font_map:
            actual_footer_ratio = font_map["footer_text"] / footer_h
            assert abs(actual_footer_ratio - expected_footer_ratio) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
