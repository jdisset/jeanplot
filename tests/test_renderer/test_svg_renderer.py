"""Tests for SVG renderer output."""

import pytest

from jeanplot import (
    Container,
    Text,
    Size,
    BoxStyle,
    Transform,
    render_to_svg,
    parse_svg,
)
from jeanplot.core.renderer.svg import SVGRenderer


class TestSVGOutput:
    """SVG string output."""

    def test_valid_svg_structure(self):
        """Output is valid SVG XML."""
        container = Container(id="box", min_dimensions=Size(100, 50))
        svg = render_to_svg(container)

        root = parse_svg(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_viewbox_set(self):
        """SVG has viewBox attribute."""
        container = Container(id="box", min_dimensions=Size(100, 50))
        svg = render_to_svg(container)

        root = parse_svg(svg)
        assert "viewBox" in root.attrib

    def test_component_id_in_output(self):
        """Component IDs appear in SVG."""
        container = Container(
            id="my_box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(background_color="#ff0000"),
        )
        svg = render_to_svg(container)

        root = parse_svg(svg)
        elem = root.find(".//*[@id='my_box']", namespaces=None)
        assert elem is not None


class TestRectangleRendering:
    """Rectangle element rendering."""

    def test_background_fill(self):
        """Background color appears as fill."""
        container = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(background_color="#00ff00"),
        )
        svg = render_to_svg(container)

        root = parse_svg(svg)
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        assert rect is not None
        # Color may have alpha appended
        fill = rect.get("fill")
        assert fill.startswith("#00ff00")

    def test_border_stroke(self):
        """Border renders as stroke."""
        container = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(
                background_color="#ffffff",
                border_color="#0000ff",
                border_width=2,
            ),
        )
        svg = render_to_svg(container)

        root = parse_svg(svg)
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        stroke = rect.get("stroke")
        assert stroke.startswith("#0000ff")
        assert float(rect.get("stroke-width")) == 2.0

    def test_corner_radius(self):
        """Corner radius renders as rx attribute."""
        container = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(background_color="#ffffff", corner_radius=10),
        )
        svg = render_to_svg(container)

        root = parse_svg(svg)
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        assert float(rect.get("rx")) == 10.0


class TestTextRendering:
    """Text element rendering."""

    def test_text_content(self, mock_renderer):
        """Text content appears in SVG."""
        parent = Container(id="parent", min_dimensions=Size(100, 50))
        text = Text(id="label", text="Hello World", font_size=12)
        parent.add_child(text)

        svg = render_to_svg(parent)
        assert "Hello World" in svg

    def test_text_attributes(self, mock_renderer):
        """Text has font attributes."""
        parent = Container(id="parent", min_dimensions=Size(100, 50))
        text = Text(
            id="label",
            text="Test",
            font_size=16,
            color="#ff0000",
        )
        parent.add_child(text)

        svg = render_to_svg(parent)
        root = parse_svg(svg)
        text_elem = root.find(".//{http://www.w3.org/2000/svg}text")
        assert text_elem is not None
        assert float(text_elem.get("font-size")) == 16.0
        assert text_elem.get("fill").startswith("#ff0000")

    def test_points_mode_scales_font_size_attribute_compensated(self):
        """Points-mode text should compensate for parent scaling in SVG output."""
        root = Container(id="root", min_dimensions=Size(200, 80))
        t1 = Text(id="t1", text="A", font_size=12, font_size_mode="points")
        scaled = Container(
            id="scaled",
            transform=Transform(scale=(2.0, 2.0)),
            children=[Text(id="t2", text="A", font_size=12, font_size_mode="points")],
        )
        root.add_child(t1)
        root.add_child(scaled)

        svg = render_to_svg(root)
        xml = parse_svg(svg)
        text1 = xml.find(".//*[@id='t1']/{http://www.w3.org/2000/svg}text")
        text2 = xml.find(".//*[@id='t2']/{http://www.w3.org/2000/svg}text")
        assert text1 is not None
        assert text2 is not None

        fs1 = text1.get("font-size")
        fs2 = text2.get("font-size")
        assert fs1 is not None and fs1.endswith("pt")
        assert fs2 is not None and fs2.endswith("pt")

        fs1_val = float(fs1[:-2])
        fs2_val = float(fs2[:-2])
        # scaled parent should get inverse-compensated font-size attr
        assert fs2_val == pytest.approx(fs1_val / 2.0, rel=1e-2)

    def test_multiline_text_uses_tspans_and_baseline(self):
        """Multiline text should emit tspans and baseline metadata."""
        parent = Container(id="parent", min_dimensions=Size(120, 60))
        text = Text(
            id="multi",
            text="line1\nline2",
            align="center",
            vertical_align="bottom",
            line_spacing=0.5,
        )
        parent.add_child(text)

        svg = render_to_svg(parent)
        xml = parse_svg(svg)
        text_elem = xml.find(".//*[@id='multi']/{http://www.w3.org/2000/svg}text")
        assert text_elem is not None
        assert text_elem.get("text-anchor") == "middle"
        assert text_elem.get("dominant-baseline") == "text-after-edge"

        tspans = text_elem.findall("{http://www.w3.org/2000/svg}tspan")
        assert len(tspans) == 2
        assert tspans[0].get("dy") == "0"
        assert tspans[1].get("dy") == "1.500em"


class TestNestedComponents:
    """Nested component rendering."""

    def test_nested_groups(self, mock_renderer):
        """Nested containers create nested groups."""
        outer = Container(
            id="outer",
            min_dimensions=Size(200, 100),
            style=BoxStyle(background_color="#cccccc"),
        )
        inner = Container(
            id="inner",
            min_dimensions=Size(100, 50),
            style=BoxStyle(background_color="#888888"),
        )
        outer.add_child(inner)

        svg = render_to_svg(outer)
        root = parse_svg(svg)

        outer_g = root.find(".//*[@id='outer']")
        inner_g = root.find(".//*[@id='inner']")
        assert outer_g is not None
        assert inner_g is not None


class TestRendererAPI:
    """SVGRenderer API."""

    def test_render_to_string(self):
        """render_to_string returns string."""
        renderer = SVGRenderer()
        container = Container(id="box", min_dimensions=Size(100, 50))
        svg = renderer.render_to_string(container)

        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_create_context(self):
        """create_context returns element."""
        renderer = SVGRenderer()
        ctx = renderer.create_context(800, 600)

        assert ctx is not None
        assert ctx.get("width") == "800"
        assert ctx.get("height") == "600"

    def test_render_to_output_file(self, tmp_path):
        """Render to file output."""
        renderer = SVGRenderer()
        container = Container(id="box", min_dimensions=Size(100, 50))
        ctx = renderer.create_context()
        renderer.render_component(ctx, container)

        output_file = tmp_path / "test.svg"
        renderer.render_to_output(ctx, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "<svg" in content
