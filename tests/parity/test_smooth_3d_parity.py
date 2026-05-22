"""Parity: jeanplot.plots.smooth_3d vs biocomp.plotutils.smooth_3d.

Both paths render the same fixture through cabinet-projected cube + smooth_2d
slices. Post-refactor the biocomp path itself delegates to jeanplot's
smooth_2d (slice kernel) but its own smooth_3d orchestrator still lives in
biocomp. These tests verify pixel-identical equivalence so biocomp's
smooth_3d can be deleted and callers migrated to jeanplot.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

pytest.importorskip("biocomp.plotutils", reason="biocomp not installed")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIXEL_TOLERANCE = 0.005


def _fixture():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    y = (x.sum(axis=1, keepdims=True) / 3 + 0.1 * rng.normal(size=(n, 1))).astype(np.float32)
    return x, y


def _diff_fraction(a_path: Path, b_path: Path) -> float:
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    arr = np.asarray(ImageChops.difference(a, b))
    return float(np.any(arr > 0, axis=-1).sum()) / (arr.shape[0] * arr.shape[1])


def _render_biocomp(X, Y, axes, **kw):
    import biocomp.plotutils as bp
    from biocomp.datautils import IdentityRescaler

    bp.smooth_3d(
        X=X, Y=Y,
        input_names=["x0", "x1", "x2"], output_name="y",
        rescaler=IdentityRescaler(), ax=axes, **kw,
    )


def _render_jeanplot(X, Y, axes, **kw):
    from jeanplot.data import IdentityRescaler
    from jeanplot.plots.smooth_3d import smooth_3d

    smooth_3d(
        X=X, Y=Y,
        input_names=["x0", "x1", "x2"], output_name="y",
        rescaler=IdentityRescaler(), ax=axes, **kw,
    )


_BASE_KW = dict(
    xlims=(0, 1), ylims=(0, 1), zlims=(0, 1),
    smooth_2d_params={
        "knn_grid_params": {
            "grid_resolution": 16,
            "knn_stats_params": {"k": 20, "radius": 0.2},
        }
    },
    show_inner_spines=False, show_front_face_ticks=True,
)


@pytest.mark.parametrize(
    "name,kw,n_axes,figsize",
    [
        ("minimal", {"zslices": [[0.2, 0.4, 0.6]], "draw_colorbar": False}, 1, (6, 6)),
        ("with_colorbar", {"zslices": [[0.2, 0.4, 0.6]], "draw_colorbar": True}, 1, (6, 6)),
        ("three_cubes", {"zslices": [[0.2], [0.4], [0.6]], "draw_colorbar": None}, 3, (18, 6)),
        (
            "otsu_contour",
            {
                "zslices": [[0.2, 0.4, 0.6]],
                "draw_colorbar": False,
                "smooth_2d_params": {
                    **_BASE_KW["smooth_2d_params"],
                    "heatmap_params": {"contours": ["otsu:0.85"]},
                },
            },
            1, (6, 6),
        ),
        (
            "title",
            {"zslices": [[0.2, 0.4, 0.6]], "draw_colorbar": False, "title": "T"},
            1, (6, 6),
        ),
    ],
)
def test_smooth_3d_pixel_parity(name, kw, n_axes, figsize, tmp_path):
    X, Y = _fixture()
    merged = {**_BASE_KW, **kw}
    if "smooth_2d_params" in kw:
        merged["smooth_2d_params"] = kw["smooth_2d_params"]

    def _make_axes():
        if n_axes == 1:
            f, a = plt.subplots(figsize=figsize, dpi=80)
            return f, [a]
        f, axs = plt.subplots(1, n_axes, figsize=figsize, dpi=80)
        return f, list(axs)

    bc_path = tmp_path / f"bc_{name}.png"
    jp_path = tmp_path / f"jp_{name}.png"

    fig, axes = _make_axes()
    _render_biocomp(X, Y, axes, **merged)
    fig.savefig(bc_path)
    plt.close(fig)

    fig, axes = _make_axes()
    _render_jeanplot(X, Y, axes, **merged)
    fig.savefig(jp_path)
    plt.close(fig)

    frac = _diff_fraction(bc_path, jp_path)
    assert frac < PIXEL_TOLERANCE, (
        f"{name}: {frac:.3%} pixel divergence (tolerance {PIXEL_TOLERANCE:.0%})"
    )
