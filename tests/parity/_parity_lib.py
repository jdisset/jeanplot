"""Shared utilities for parity tests.

Lives alongside conftest.py; conftest.py inserts this directory into sys.path
so sibling test files can `from _parity_lib import ...`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from jeanplot import PlotData


PARITY_DIR = Path(__file__).parent
FIXTURES_DIR = PARITY_DIR / "fixtures"
JOBS_DIR = PARITY_DIR / "jeanplot_jobs"
BASELINES_DIR = PARITY_DIR / "baselines"

PIXEL_TOLERANCE = 0.02
DPI = 80
FIG_SIZE = (4.0, 4.0)


def load_fixture(name: str) -> PlotData:
    data = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    return PlotData(
        xval=np.asarray(data["x"], dtype=np.float32),
        yval=np.asarray(data["y"], dtype=np.float32),
        input_names=data["input_names"],
        output_name=data.get("output_name", "output"),
        metadata=data.get("metadata", {}),
    )


def load_mvp_fixture() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((FIXTURES_DIR / "mvp_pair.json").read_text())
    return (
        np.asarray(data["measured"], dtype=np.float32),
        np.asarray(data["predicted"], dtype=np.float32),
    )


def diff_fraction(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        return 1.0
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    arr = np.asarray(diff)
    return float(np.any(arr > 0, axis=-1).sum()) / (arr.shape[0] * arr.shape[1])
