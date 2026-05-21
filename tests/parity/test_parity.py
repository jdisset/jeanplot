"""Parity / regression tests for jeanplot-plot.

Each fixture is rendered via a YAML job (the same shape jeanplot-plot
consumes from the CLI) and the resulting PNG is compared to a checked-in
baseline within a per-pixel tolerance.

On first run, baselines are auto-created and the test xfails so the
developer notices and commits them. Subsequent runs assert tolerance.
"""

from pathlib import Path

import matplotlib
import pytest
from PIL import Image

matplotlib.use("Agg")

import dracon as dr

import jeanplot
from jeanplot import Figure, PlotData, Size, make_plot_context

from _parity_lib import (
    BASELINES_DIR,
    DPI,
    FIG_SIZE,
    JOBS_DIR,
    PIXEL_TOLERANCE,
    diff_fraction,
    load_fixture,
    load_mvp_fixture,
)


PARITY_FIXTURES = ["1d_smooth", "2d_smooth", "3d_smooth", "mvp_pair"]


def _fixture_context(name: str) -> dict:
    if name == "mvp_pair":
        m, p = load_mvp_fixture()
        return {"measured": m, "predicted": p}
    return {"plot_data": load_fixture(name)}


def _render(job_yaml: Path, context_extra: dict, out_path: Path) -> Figure:
    cfg = dr.load(
        str(job_yaml),
        enable_interpolation=True,
        raw_dict=True,
        context={**make_plot_context(), **context_extra},
    )
    fig: Figure = cfg["figure"]
    fig.min_dimensions = Size(width=FIG_SIZE[0], height=FIG_SIZE[1])
    fig.dpi = DPI
    fig.output_dir = str(out_path.parent)
    fig.output_file = None  # render in-memory, side-step mpl's strict png metadata
    mfig = jeanplot.render(fig, overwrite=True)
    if mfig is not None:
        mfig.savefig(out_path, dpi=DPI)
        import matplotlib.pyplot as plt

        plt.close(mfig)
    return fig


@pytest.mark.parametrize("fixture", PARITY_FIXTURES)
def test_jeanplot_self_parity(fixture, tmp_path):
    job_yaml = JOBS_DIR / f"{fixture}.yaml"
    out_png = tmp_path / f"{fixture}.png"
    _render(job_yaml, _fixture_context(fixture), out_png)
    assert out_png.exists(), f"render did not write {out_png}"

    baseline = BASELINES_DIR / f"{fixture}.png"
    if not baseline.exists():
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        Image.open(out_png).save(baseline)
        pytest.xfail(
            f"baseline missing for {fixture} — wrote one at {baseline}; commit it and re-run"
        )

    frac = diff_fraction(Image.open(baseline), Image.open(out_png))
    assert frac < PIXEL_TOLERANCE, (
        f"{fixture}: {frac:.3%} of pixels differ from baseline (tolerance {PIXEL_TOLERANCE:.0%})"
    )


def test_fixture_loader_returns_plot_data():
    pd = load_fixture("2d_smooth")
    assert isinstance(pd, PlotData)
    assert pd.x.shape == (200, 2)
    assert pd.y.shape == (200, 1)
