from importlib.resources import files, as_file
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_from_pkg(path: str, pkg: str | None = "jeanplot"):
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
    except FileNotFoundError:
        return None
