from typing import Optional
from importlib.resources import files, as_file
from pathlib import Path
import logging
from .models import AnchorPoint, Offset

logger = logging.getLogger(__name__)


def read_from_pkg(path: str, pkg: Optional[str] = "jeanplot"):
    """
    reads a file from a python package with syntax
    package.module:path/to/file.txt
    """

    if ":" in path:
        fpkg, fpath = path.split(":", maxsplit=1)
        fpath = Path(fpath)
        assert fpkg == pkg, f"Package mismatch: {fpkg} != {pkg}"
    else:
        fpath = Path(path)
        fpkg = pkg

    if not fpkg:
        raise ValueError(f"No package specified when asking for {path}.")

    try:
        with as_file(files(fpkg) / fpath.as_posix()) as p:
            with open(p, "r") as f:
                return f.read()
    except FileNotFoundError:
        pass

    # it failed
    resources = [resource.name for resource in files(fpkg).iterdir() if not resource.is_file()]
    resources_str = "\n  - ".join(resources)
    msg = f"""File not found in package {fpkg}: {fpath}.
        Package root: {files(fpkg)}
        Available subdirs:
        - {resources_str}"""

    raise FileNotFoundError(msg)


def load_file(path: str | Path):
    """reads a file that can be in a package or in the filesystem. if in a package it'll start with pkg:"""
    if isinstance(path, Path):
        path = path.as_posix()
    if path.startswith("pkg:"):
        return read_from_pkg(path[4:])
    else:
        with open(path, "r") as f:
            return f.read()


def load_file_if_exists(path: str | Path):
    """reads a file that can be in a package or in the filesystem. if in a package it'll start with pkg:"""
    try:
        return load_file(path)
    except FileNotFoundError as e:
        return None


def make_anchor_point(
    position: str, min_segment: float = 10.0, offset: Optional[Offset] = None
) -> AnchorPoint:
    """create anchor at standard position"""
    directions = {
        "top": (0, -1),
        "bottom": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
        "top_left": (-0.707, -0.707),
        "top_right": (0.707, -0.707),
        "bottom_left": (-0.707, 0.707),
        "bottom_right": (0.707, 0.707),
    }

    offsets = {
        "top": Offset(relative=(0.5, 0)),
        "bottom": Offset(relative=(0.5, 1)),
        "left": Offset(relative=(0, 0.5)),
        "right": Offset(relative=(1, 0.5)),
        "top_left": Offset(relative=(0, 0)),
        "top_right": Offset(relative=(1, 0)),
        "bottom_left": Offset(relative=(0, 1)),
        "bottom_right": Offset(relative=(1, 1)),
    }

    return AnchorPoint(
        offset=offset or offsets.get(position, Offset(relative=(0.5, 0.5))),
        direction=directions.get(position, (0, 1)),
        min_segment=min_segment,
    )


def standard_anchors(min_segment: float = 10.0) -> list[AnchorPoint]:
    """get standard anchor points"""
    return [
        make_anchor_point("top", min_segment),
        make_anchor_point("right", min_segment),
        make_anchor_point("bottom", min_segment),
        make_anchor_point("left", min_segment),
    ]
