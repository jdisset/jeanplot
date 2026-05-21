"""Tests for theme color_remap interpolation bug.

Regression test for: when color_remap keys are quoted in YAML
(e.g. "${theme.input_main_marker_color}"), Dracon must interpolate them so
the resulting cascade carries hex color keys, not literal interpolation
strings that fail color normalization.
"""

from jeanplot import jstyle, SVGElement
from jeanplot.gene.elements import FluoMarker
from jeanplot.gene.schematic import SourceAnnotation


def test_svg_element_color_remap_validator_handles_interpolation_strings():
    """SVGElement validator should skip invalid color strings gracefully."""
    svg = SVGElement(
        id="test",
        color_remap={
            "${theme.input_main_marker_color}": "#aaaaaa",
            "#0000ff": "#aaaaaa",
        }
    )
    assert "${theme.input_main_marker_color}" not in svg.color_remap
    assert any(k.startswith("#0000ff") for k in svg.color_remap.keys())


def test_default_theme_resolves_color_remap_keys_on_real_components():
    """After load_default_theme + apply, color_remap keys on a FluoMarker's
    SVGElement child are real hex colors (not unresolved interpolation strings)."""
    from jeanplot import load_default_theme
    load_default_theme()

    fm = FluoMarker(id="m1", part_name="MKO2")
    svg = SVGElement(id="svg1")
    fm.add_child(svg)
    jstyle.apply(fm)

    assert svg.color_remap, "expected color_remap to be populated by the theme"
    for key in svg.color_remap.keys():
        assert not str(key).startswith("${"), (
            f"color_remap key is an unresolved interpolation string: {key}"
        )
        assert str(key).startswith("#"), (
            f"color_remap key should be a hex color: {key}"
        )


def test_default_theme_resolves_source_annotation_colors():
    """SourceAnnotation styles in the default theme resolve to hex colors."""
    from jeanplot import load_default_theme
    load_default_theme()

    sa = SourceAnnotation(id="sa1", marker="MKO2")
    jstyle.apply(sa)

    assert sa.style is not None
    assert sa.style.border_color is not None
    assert str(sa.style.border_color).startswith("#"), (
        f"border_color should be hex: {sa.style.border_color}"
    )


def test_load_default_theme_replaces_custom_styles():
    """Calling load_default_theme should restore defaults, not keep ad-hoc styles."""
    from jeanplot import load_default_theme

    jstyle.clear()
    jstyle.update({"Text": {"color": "red"}})
    assert jstyle._cascade is not None
    custom_tree = jstyle._cascade._rule_tree

    load_default_theme()
    assert jstyle._cascade is not None
    loaded_tree = jstyle._cascade._rule_tree
    assert loaded_tree is not custom_tree
    assert len(loaded_tree) > 1, "default theme should carry many rules"
