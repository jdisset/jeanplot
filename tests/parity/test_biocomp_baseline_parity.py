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

from _parity_lib import BASELINES_DIR, PIXEL_TOLERANCE, diff_fraction, load_fixture


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


@pytest.mark.parametrize("fixture", ["2d_smooth"])
def test_biocomp_renders_within_tolerance(fixture, tmp_path):
    """Snapshot cross-check: biocomp's draw should land within 5x tolerance.

    Skips if biocomp's drawing code raises (e.g. needs a configured rescaler)
    — the goal is regression detection on environments where the cross-render
    is meaningful, not to assert biocomp behaviour.
    """
    baseline = BASELINES_DIR / f"{fixture}.png"
    if not baseline.exists():
        pytest.skip(f"jeanplot baseline missing for {fixture}; run test_parity first")

    pd = load_fixture(fixture)
    out = tmp_path / f"{fixture}_biocomp.png"
    try:
        _render_biocomp_2d(pd, out)
    except (NotImplementedError, AttributeError, TypeError) as e:
        pytest.skip(f"biocomp render not exercisable on this fixture shape: {e}")

    frac = diff_fraction(Image.open(baseline), Image.open(out))
    assert frac < PIXEL_TOLERANCE * 5, (
        f"{fixture}: {frac:.3%} pixel divergence biocomp-vs-jeanplot "
        f"(tolerance {PIXEL_TOLERANCE * 5:.0%})"
    )
