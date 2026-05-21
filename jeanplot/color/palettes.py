import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import dracon as dr


def load_palettes(path: str) -> dict:
    """Load a YAML palette file and return its parsed contents.

    Returns a dict with at least `color_maps: {name: [hex, ...]}`. The
    `default_color_map` key (if present) is preserved on the returned dict.
    """
    return dr.resolve_all_lazy(dr.load(path))


def register_palettes(palettes: dict) -> dict[str, mcolors.LinearSegmentedColormap]:
    """Register every entry in `palettes['color_maps']` as a matplotlib cmap.

    Idempotent: unregisters then re-registers existing names.
    Returns the dict of constructed cmaps.
    """
    cmap_defs = palettes.get("color_maps") or {}
    cmaps = {
        name: mcolors.LinearSegmentedColormap.from_list(name, colors, N=256)
        for name, colors in cmap_defs.items()
    }
    registered = plt.colormaps()
    for name, cmap in cmaps.items():
        if name in registered:
            plt.colormaps.unregister(name)
        plt.colormaps.register(cmap, name=name)
    return cmaps


def default_cmap_name(palettes: dict, fallback: str = "viridis") -> str:
    return palettes.get("default_color_map") or fallback
