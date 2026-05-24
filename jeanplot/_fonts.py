"""Register bundled fonts (jeanplot/resources/fonts/) with matplotlib so the
default theme's font chain resolves without system installation. Ships Poppins
+ Roboto (SIL OFL 1.1); OFL.txt files sit alongside the .ttf payloads."""

import logging
from importlib.resources import files
from pathlib import Path

import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_bundled_fonts(force: bool = False) -> list[Path]:
    """Add bundled *.ttf/*.otf to matplotlib's font manager. Idempotent unless force."""
    global _REGISTERED
    if _REGISTERED and not force:
        return []

    root = Path(str(files("jeanplot").joinpath("resources/fonts")))
    if not root.is_dir():
        logger.warning("jeanplot bundled fonts dir missing: %s", root)
        _REGISTERED = True
        return []

    paths = sorted(root.rglob("*.ttf")) + sorted(root.rglob("*.otf"))
    for p in paths:
        try:
            fm.fontManager.addfont(str(p))
        except Exception as exc:
            logger.warning("Failed to register bundled font %s: %s", p.name, exc)

    _REGISTERED = True
    return paths
