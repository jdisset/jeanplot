"""Tests for theme color_remap interpolation bug.

Regression test for: https://github.com/anthropics/claude-code/issues/XXX
When color_remap keys are quoted in YAML (e.g., "${theme.input_main_marker_color}"),
Dracon doesn't interpolate them, leaving literal strings that fail color normalization.
"""

import pytest
import dracon as dr

from jeanplot import jstyle, make_context_from_types, DEFAULT_TYPES, SVGElement


def test_theme_color_remap_keys_are_resolved():
    """Color_remap keys should be hex colors, not unresolved interpolation strings."""
    _DEFAULT_THEME_PATH = "pkg:jeanplot:resources/themes/default.yaml"

    theme_dict = dr.load(
        _DEFAULT_THEME_PATH,
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES),
    )
    dr.resolve_all_lazy(theme_dict)

    # check GeneticPart > SVGElement > color_remap
    genetic_part_styles = theme_dict.get("GeneticPart", {})
    svg_element_styles = genetic_part_styles.get("SVGElement", {})
    color_remap = svg_element_styles.get("color_remap", {})

    # verify no unresolved interpolations in keys
    for key in color_remap.keys():
        assert not key.startswith("${"), (
            f"color_remap key '{key}' is an unresolved interpolation string. "
            "Keys should be hex colors like '#0000ff', not '${theme.xxx}'."
        )
        # keys should be valid hex colors (starting with #)
        assert key.startswith("#"), (
            f"color_remap key '{key}' should be a hex color starting with '#'"
        )


def test_svg_element_color_remap_validator_handles_interpolation_strings():
    """SVGElement validator should skip invalid color strings gracefully."""
    # this test verifies the current behavior - warnings are logged but no crash
    svg = SVGElement(
        id="test",
        color_remap={
            "${theme.input_main_marker_color}": "#aaaaaa",  # invalid key
            "#0000ff": "#aaaaaa",  # valid key
        }
    )
    # invalid key should be silently dropped, valid key should be normalized
    assert "${theme.input_main_marker_color}" not in svg.color_remap
    # valid key should be present (normalized to hex with alpha)
    assert any(k.startswith("#0000ff") for k in svg.color_remap.keys())


def test_jstyle_loaded_theme_has_resolved_colors():
    """jstyle _raw_styles should have resolved color values after theme load."""
    # explicitly load theme (may be cleared by fixture)
    from jeanplot import load_default_theme
    load_default_theme()

    # look for any color_remap in jstyle's raw styles
    def find_color_remaps(d, path=''):
        results = []
        if isinstance(d, dict):
            if 'color_remap' in d:
                results.append((path + '.color_remap', d['color_remap']))
            for k, v in d.items():
                results.extend(find_color_remaps(v, path + '.' + k if path else k))
        return results

    remaps = find_color_remaps(jstyle._raw_styles)
    assert len(remaps) > 0, "Expected to find color_remap in jstyle._raw_styles"

    # verify no unresolved interpolation in any color_remap keys
    for path, remap in remaps:
        for key in remap.keys():
            assert not key.startswith("${"), (
                f"jstyle._raw_styles has unresolved color_remap key at {path}: {key}"
            )
