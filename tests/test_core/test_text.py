"""Tests for Text component."""
import matplotlib.pyplot as plt
import pytest

from jeanplot import Text, Container, Size, render_to_svg, parse_svg
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer


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

    def test_text_anchor_alignment_right(self, mock_renderer):
        """Right text alignment maps to SVG end anchor."""
        parent = Container(id="p", min_dimensions=Size(100, 50))
        t = Text(id="lbl", text="Hello", align="right")
        parent.add_child(t)

        svg = render_to_svg(parent)
        root = parse_svg(svg)
        text_elem = root.find(".//{http://www.w3.org/2000/svg}text")
        assert text_elem is not None
        assert text_elem.get("text-anchor") == "end"


class TestMatplotlibTextAlignment:
    """Headless placement checks for matplotlib text alignment."""

    @staticmethod
    def _render_single_text(
        align: str = "left", vertical_align: str = "middle"
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], str, str]:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 120)
        ax.set_aspect("equal")
        ax.axis("off")

        text = Text(
            id="probe",
            text="ALIGN",
            font_size=10,
            align=align,  # type: ignore[arg-type]
            vertical_align=vertical_align,  # type: ignore[arg-type]
            min_dimensions=Size(120, 30),
        )
        parent = Container(id="p", min_dimensions=Size(120, 30), children=[text])

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, parent, adjust_lims=False)
        fig.canvas.draw()

        artist = next(a for a in ax.texts if a.get_text() == "ALIGN")
        bbox = artist.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        comp_bounds = text.get_world_bounds()
        assert comp_bounds is not None

        plt.close(fig)
        return comp_bounds, (x0, y0, x1, y1), artist.get_ha(), artist.get_va()

    @pytest.mark.parametrize("align,expected_ha", [("left", "left"), ("center", "center"), ("right", "right")])
    def test_horizontal_alignment_matches_component_bounds(self, align, expected_ha):
        comp, txt, ha, _va = self._render_single_text(align=align, vertical_align="middle")
        cx0, _cy0, cx1, _cy1 = comp
        tx0, _ty0, tx1, _ty1 = txt
        assert ha == expected_ha
        if align == "left":
            assert abs(tx0 - cx0) < 1.0
        elif align == "center":
            assert abs(((tx0 + tx1) / 2.0) - ((cx0 + cx1) / 2.0)) < 1.0
        else:
            assert abs(tx1 - cx1) < 1.0

    @pytest.mark.parametrize(
        "vertical_align,expected_va",
        [("top", "top"), ("middle", "center"), ("bottom", "bottom")],
    )
    def test_vertical_alignment_matches_component_bounds(self, vertical_align, expected_va):
        comp, txt, _ha, va = self._render_single_text(align="center", vertical_align=vertical_align)
        _cx0, cy0, _cx1, cy1 = comp
        _tx0, ty0, _tx1, ty1 = txt
        assert va == expected_va
        if vertical_align == "top":
            assert abs(ty1 - cy1) < 1.5
        elif vertical_align == "middle":
            assert abs(((ty0 + ty1) / 2.0) - ((cy0 + cy1) / 2.0)) < 1.0
        else:
            assert abs(ty0 - cy0) < 1.5
