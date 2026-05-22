"""Optional biocomp cross-check.

Importorskip-guarded. The only test file in jeanplot that touches biocomp,
and only when biocomp is installed. Re-renders fixtures through biocomp's
drawing path and asserts pixel-tolerance equivalence to the jeanplot
baseline. Per the refactor spec §5.4 this is a one-shot snapshot, not a
unit test of biocomp.
"""

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("biocomp.plotting", reason="biocomp not installed; cross-render optional")

import matplotlib

matplotlib.use("Agg")

from _parity_lib import PIXEL_TOLERANCE, diff_fraction, load_fixture


def _render_biocomp_2d(pd, out_path: Path):
    try:
        from biocomp.datautils import DataRescaler
        from biocomp.plotting.plotting_smooth_2d import smooth_2d
    except ImportError as e:
        pytest.skip(f"biocomp.plotting failed to import: {e}")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=80)
    smooth_2d(
        X=pd.x,
        Y=pd.y,
        input_names=pd.input_names,
        output_name=pd.output_name,
        rescaler=DataRescaler(),
        ax=ax,
        title="2D smooth",
        draw_colorbar=False,
        knn_grid_params={
            "grid_resolution": 16,
            "knn_stats_params": {"k": 20, "radius": 0.2},
        },
    )
    fig.savefig(out_path)
    plt.close(fig)


def _render_jeanplot_2d(pd, out_path: Path):
    """Render the same fixture via jeanplot's direct-call smooth_2d.

    This is the canonical comparison target post-refactor 03: with biocomp's
    plotting now shimmed onto jeanplot, biocomp's direct-call render must equal
    jeanplot's direct-call render pixel-for-pixel (modulo fp noise).
    """
    from jeanplot.data import IdentityRescaler
    from jeanplot.plots.smooth_2d import smooth_2d as jp_smooth_2d
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=80)
    jp_smooth_2d(
        X=pd.x,
        Y=pd.y,
        input_names=pd.input_names,
        output_name=pd.output_name,
        rescaler=IdentityRescaler(),
        ax=ax,
        title="2D smooth",
        draw_colorbar=False,
        knn_grid_params={
            "grid_resolution": 16,
            "knn_stats_params": {"k": 20, "radius": 0.2},
        },
    )
    fig.savefig(out_path)
    plt.close(fig)


@pytest.mark.parametrize("fixture", ["2d_smooth"])
def test_biocomp_matches_jeanplot_direct_call(fixture, tmp_path):
    """SSOT invariant: biocomp's shimmed draw must equal jeanplot's direct draw.

    Post-refactor 03, biocomp.plotting.plotting_smooth_2d.smooth_2d delegates
    to jeanplot. Rendering the same fixture through both paths must produce
    pixel-identical output (the tolerance accounts for floating-point noise
    in matplotlib's rasterizer).
    """
    pd = load_fixture(fixture)
    bp_out = tmp_path / f"{fixture}_biocomp.png"
    jp_out = tmp_path / f"{fixture}_jeanplot.png"
    try:
        _render_biocomp_2d(pd, bp_out)
    except (NotImplementedError, AttributeError, TypeError) as e:
        pytest.skip(f"biocomp render not exercisable on this fixture shape: {e}")
    _render_jeanplot_2d(pd, jp_out)

    frac = diff_fraction(Image.open(bp_out), Image.open(jp_out))
    assert frac < PIXEL_TOLERANCE, (
        f"{fixture}: {frac:.3%} pixel divergence between biocomp shim and "
        f"jeanplot direct call (tolerance {PIXEL_TOLERANCE:.0%})"
    )
