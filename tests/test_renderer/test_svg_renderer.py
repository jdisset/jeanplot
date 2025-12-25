"""Tests for SVG renderer output."""

from jeanplot import (
    Container,
    Text,
    Size,
    BoxStyle,
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
