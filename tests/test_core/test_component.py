"""Tests for Component base class."""
from jeanplot import Component, Container, Size, AnchorComponent, BoxStyle


def test_component_default_dimensions():
    """Component has zero default dimensions."""
    c = Component(id="c")
    assert c.min_dimensions == Size(width=0, height=0)


def test_component_id_assignment():
    """ID assigned and accessible."""
    c = Component(id="my_comp")
    assert c.id == "my_comp"


def test_component_parent_link():
    """Parent-child relationship established."""
    parent = Container(id="parent")
    child = Component(id="child")
    parent.add_child(child)
    assert child.parent is parent
    assert child in parent.children


def test_component_show_default_true():
    """Components visible by default."""
    c = Component()
    assert c.show is True


def test_component_show_false_not_rendered(mock_renderer):
    """Hidden component not rendered to SVG."""
    from jeanplot import render_to_svg

    c = Container(
        id="hidden",
        min_dimensions=Size(100, 100),
        show=False,
        style=BoxStyle(background_color="#ff0000"),
    )
    c.measure_and_layout(mock_renderer)
    svg = render_to_svg(c)
    # Hidden component shouldn't produce visible rect
    assert "hidden" not in svg or 'fill="none"' in svg


def test_anchor_direction():
    """AnchorComponent stores direction vector."""
    anchor = AnchorComponent(id="a", direction=(0, 1))
    assert anchor.direction == (0, 1)


def test_anchor_default_direction_is_none():
    """Default anchor has no direction."""
    anchor = AnchorComponent(id="a")
    assert anchor.direction is None


def test_component_style_class_list():
    """Style class stored as list."""
    c = Component(id="c", style_class=["primary", "large"])
    assert "primary" in c.style_class
    assert "large" in c.style_class
