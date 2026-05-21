import matplotlib.pyplot as plt
import pytest

from jeanplot.color import load_palettes, register_palettes, default_cmap_name, closest_name


@pytest.fixture
def palettes():
    return load_palettes("pkg:jeanplot:resources/colors/bio_palettes.yaml")


def test_palette_keys(palettes):
    assert "color_maps" in palettes
    assert "bc_blues" in palettes["color_maps"]
    assert isinstance(palettes["color_maps"]["bc_blues"], list)


def test_register_palettes_makes_cmaps_available(palettes):
    register_palettes(palettes)
    assert "bc_blues" in plt.colormaps()
    assert "bc_reds" in plt.colormaps()


def test_register_is_idempotent(palettes):
    register_palettes(palettes)
    register_palettes(palettes)
    assert "bc_blues" in plt.colormaps()


def test_default_cmap_name(palettes):
    assert default_cmap_name(palettes) == "bc_blues"
    assert default_cmap_name({}, fallback="plasma") == "plasma"


def test_closest_name_matches_case_insensitively():
    assert closest_name("EBFP", ["ebfp", "eyfp", "mkate"]) == "ebfp"


def test_closest_name_default_on_miss():
    assert closest_name("zzzzz", ["abc", "def"], default="abc") == "abc"
    assert closest_name("zzzzz", ["abc", "def"]) is None
