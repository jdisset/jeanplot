"""Tests for Text component."""
from jeanplot import Text, Container, Size, render_to_svg, parse_svg


class TestTextComponent:
    """Text component basics."""

    def test_text_content(self):
        """Text stores content string."""
        t = Text(id="t", text="Hello")
        assert t.text == "Hello"

    def test_font_size_default(self):
        """Default font size set."""
        t = Text(id="t", text="Hello")
        assert t.font_size > 0

    def test_font_properties(self):
        """Font properties stored."""
        t = Text(
            id="t",
            text="Hello",
            font_size=24,
            font_name="Arial",
            font_weight="bold",
        )
        assert t.font_size == 24
        assert t.font_name == "Arial"
        assert t.font_weight == "bold"

    def test_color_default(self):
        """Default text color is black."""
        t = Text(id="t", text="Hello")
        assert t.color == "black" or (t.color and "000" in t.color)


class TestTextLayout:
    """Text layout behavior."""

    def test_text_has_dimensions(self, mock_renderer):
        """Text has dimensions after layout."""
        parent = Container(id="p", min_dimensions=Size(200, 50))
        t = Text(id="t", text="Hello World", font_size=12)
        parent.add_child(t)
        parent.measure_and_layout(mock_renderer)
        assert t._dimensions.width > 0
        assert t._dimensions.height > 0

    def test_multiline_text_taller(self, mock_renderer):
        """Multiline text is taller."""
        single = Container(id="p1", min_dimensions=Size(200, 50))
        multi = Container(id="p2", min_dimensions=Size(200, 100))
        t1 = Text(id="t1", text="Line", font_size=12)
        t2 = Text(id="t2", text="Line1\nLine2\nLine3", font_size=12)
        single.add_child(t1)
        multi.add_child(t2)

        single.measure_and_layout(mock_renderer)
        multi.measure_and_layout(mock_renderer)

        assert t2._dimensions.height > t1._dimensions.height


class TestTextRendering:
    """Text SVG rendering."""

    def test_text_in_svg(self, mock_renderer):
        """Text renders to SVG."""
        parent = Container(id="p", min_dimensions=Size(100, 50))
        t = Text(id="lbl", text="Hello")
        parent.add_child(t)

        svg = render_to_svg(parent)
        assert "Hello" in svg

    def test_text_anchor_alignment(self, mock_renderer):
        """Text alignment affects anchor."""
        parent = Container(id="p", min_dimensions=Size(100, 50))
        t = Text(id="lbl", text="Hello", align="center")
        parent.add_child(t)

        svg = render_to_svg(parent)
        root = parse_svg(svg)
        text_elem = root.find(".//{http://www.w3.org/2000/svg}text")
        assert text_elem is not None
        assert text_elem.get("text-anchor") == "middle"
