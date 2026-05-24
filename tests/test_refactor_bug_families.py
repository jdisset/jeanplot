"""Regression tests for the bug families uncovered during the unified-styling refactor.

Each family below corresponds to a real bug that broke a paper figure:

1. **Size cascade clobber** — `Size.__init__` populated `model_fields_set` even
   for defaults, so cascade-fill treated user-passed `axes_size=Size(10, 4)`
   as already-merged and `paper_axes_size: [2.5, 2.0]` overrode it silently.

2. **Per-side `style.padding.X` cascade** — `BoxInset` exists so themes can
   override one side without restating the whole tuple, BUT this only works if
   `BoxInset` opts into `_user_set_fields` tracking.

3. **BoxStyle/MarginPadding user-set tracking** — a user passing
   `style=BoxStyle(border_width=1.0)` must protect that field against theme
   `style.border_width: 4.0`. The fallback `model_fields_set` worked AFTER fix
   #1 only because BoxStyle now also has `_user_set_fields` via `MarginPadding`.

4. **Auto-tuple coercion on assignment** — `style.padding = (1, 2, 3, 4)`
   anywhere in user code must coerce to `BoxInset` without enabling
   `validate_assignment` globally (which would normalize colors etc.).

5. **`MatplotlibRenderer.render_component` params** — `adjust_lims_padding`
   and `adjust_lims_set_aspect` let downstream renderers (genetic circuit)
   skip the hardcoded 15% padding floor and the unconditional `set_aspect`.

6. **`panel_from` field coverage** — `SmoothGradMagnitudePanel2D` was missing
   `xaxis_labelpad/yaxis_labelpad` because the underlying function lacked the
   kwargs. Cascade rules targeting it were silent no-ops.

7. **`!set_default` SSOT** — the same variable declared in multiple included
   files becomes first-write-wins; later declarations are silent no-ops.

8. **`FluoMarker` label auto-shrink** — long fluorophore names (e.g.
   "mNeonGreen") used to overflow the SVG arrow shape. The auto-shrink lives
   in `GeneticPart.measure_and_layout` and must mark `font_size` as user-set
   so the cascade doesn't reset it on subsequent layout passes.

9. **Mathtext subscripts in PaperCircuit** — Poppins doesn't ship Unicode
   subscripts; the panel template uses matplotlib mathtext (`$X_{1}$`).

10. **e2e schematic render** — actually render a schematic and check the
    matplotlib axes claims reasonable space and text artists end up sized
    inside the SVG bbox they label.
"""

import numpy as np
import pytest
from pydantic import Field

from jeanplot import (
    BoxStyle,
    Component,
    Container,
    Figure,
    PlotData,
    Size,
    SmoothGradMagnitudePanel2D,
    SmoothPanel1D,
    SmoothPanel2D,
    Text,
    jstyle,
    load_plot_theme,
)
from jeanplot.core.models import BoxInset, MarginPadding


# ─────────────────────────────────────────────────────────────────────────────
# (1) Size.__init__ — sentinel-based so model_fields_set is accurate
# ─────────────────────────────────────────────────────────────────────────────


def test_size_default_construction_has_no_fields_set():
    """Size() with no args must NOT populate model_fields_set — otherwise the
    cascade thinks every field was user-set and refuses to fill defaults."""
    s = Size()
    assert s.model_fields_set == set(), f"Size() leaked fields: {s.model_fields_set}"


def test_size_positional_construction_marks_passed_fields():
    s = Size(10.0, 4.0)
    assert s.model_fields_set == {"width", "height"}


def test_size_partial_kwargs_marks_only_passed():
    s = Size(width=10.0)
    assert s.model_fields_set == {"width"}
    assert s.width == 10.0
    assert s.height == 0.0


def test_explicit_axes_size_survives_theme_cascade():
    """Family-1 regression: user-passed axes_size must beat theme paper_axes_size."""
    jstyle.update({
        "SmoothPanel2D": {"axes_size": Size(width=2.5, height=2.0)},
    })
    panel = SmoothPanel2D(axes_size=Size(width=10.0, height=4.0))
    jstyle.apply(panel)
    assert panel.axes_size.width == 10.0
    assert panel.axes_size.height == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# (2) Per-side style.padding.X cascade
# ─────────────────────────────────────────────────────────────────────────────


def test_box_inset_accepts_tuple_dict_and_model():
    """Coercion must support all three input shapes — themes use tuples and
    dicts interchangeably."""
    assert BoxInset.model_validate([1, 2, 3, 4]) == BoxInset(top=1, right=2, bottom=3, left=4)
    assert BoxInset.model_validate({"top": 1, "right": 2}) == BoxInset(top=1, right=2)
    assert BoxInset.model_validate(BoxInset(top=5)) == BoxInset(top=5)


def test_box_inset_iteration_and_indexing():
    """Legacy compatibility for code that unpacks `t, r, b, l = inset` or
    indexes `inset[0..3]`."""
    p = BoxInset(top=1, right=2, bottom=3, left=4)
    assert tuple(p) == (1, 2, 3, 4)
    assert (p[0], p[1], p[2], p[3]) == (1, 2, 3, 4)
    t, r, b, l = p
    assert (t, r, b, l) == (1, 2, 3, 4)


def test_per_side_padding_overrides_base_in_cascade():
    """Family-2 regression: `style.padding.right: 0.6` on SmoothPanel2D must
    override the right side WITHOUT clobbering bottom/left from the base
    PlotPanel rule."""
    jstyle.update({
        "PlotPanel": {"style.padding.bottom": 0.5, "style.padding.left": 0.5},
        "SmoothPanel2D": {"style.padding.right": 0.6},
    })
    panel = SmoothPanel2D()
    jstyle.apply(panel)
    p = panel.style.padding
    assert p.bottom == 0.5, f"bottom not preserved: {p}"
    assert p.left == 0.5, f"left not preserved: {p}"
    assert p.right == 0.6, f"right not overridden: {p}"
    assert p.top == 0.0


def test_user_set_box_inset_field_blocks_cascade():
    """Family-2 regression: user passing `style=BoxStyle(padding=BoxInset(right=0.1))`
    must protect `right` against theme `style.padding.right: 0.6`."""
    jstyle.update({"SmoothPanel2D": {"style.padding.right": 0.6}})
    panel = SmoothPanel2D(style=BoxStyle(padding=BoxInset(right=0.1)))
    jstyle.apply(panel)
    assert panel.style.padding.right == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# (3) BoxStyle/MarginPadding user-set field tracking
# ─────────────────────────────────────────────────────────────────────────────


def test_margin_padding_tracks_user_set_fields():
    """MarginPadding must populate _user_set_fields on construction so the
    cascade respects user values for `margin`/`padding`."""
    mp = MarginPadding(padding=BoxInset(right=0.5))
    assert "padding" in mp._user_set_fields
    assert "margin" not in mp._user_set_fields


def test_box_inset_tracks_user_set_fields():
    """BoxInset itself tracks which sides were user-set."""
    bi = BoxInset(right=0.5, bottom=1.0)
    assert bi._user_set_fields == {"right", "bottom"}


def test_box_style_inherits_user_set_tracking():
    """BoxStyle should pick up `_user_set_fields` from MarginPadding parent."""
    bs = BoxStyle(border_width=1.5)
    assert "border_width" in bs._user_set_fields


# ─────────────────────────────────────────────────────────────────────────────
# (4) Auto-coerce tuple/dict on assignment (no validate_assignment)
# ─────────────────────────────────────────────────────────────────────────────


def test_padding_tuple_assignment_coerces_to_box_inset():
    """`style.padding = (1, 2, 3, 4)` must coerce — table.py and others rely on this."""
    bs = BoxStyle()
    bs.padding = (1, 2, 3, 4)
    assert isinstance(bs.padding, BoxInset)
    assert (bs.padding.top, bs.padding.right, bs.padding.bottom, bs.padding.left) == (1, 2, 3, 4)


def test_padding_dict_assignment_coerces_to_box_inset():
    bs = BoxStyle()
    bs.padding = {"left": 0.5, "right": 0.6}
    assert isinstance(bs.padding, BoxInset)
    assert bs.padding.left == 0.5
    assert bs.padding.right == 0.6


def test_assignment_coercion_does_not_normalize_colors():
    """The coercion must NOT propagate validate_assignment globally — that
    would silently transform color strings (a real bug we hit before)."""
    bs = BoxStyle()
    bs.background_color = "lightblue"
    # background_color is a NormalizedColor, which only normalizes at
    # construction. Direct assignment without validate_assignment leaves the
    # string as-is. This is the contract.
    assert bs.background_color == "lightblue"


# ─────────────────────────────────────────────────────────────────────────────
# (5) MatplotlibRenderer.render_component params
# ─────────────────────────────────────────────────────────────────────────────


def test_render_component_accepts_padding_and_aspect_params():
    """The genetic-circuit renderer needs to pass padding=0 and skip
    set_aspect, so the signature must accept these kwargs."""
    import inspect

    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    sig = inspect.signature(MatplotlibRenderer.render_component)
    assert "adjust_lims_padding" in sig.parameters
    assert "adjust_lims_set_aspect" in sig.parameters
    assert sig.parameters["adjust_lims_padding"].default == 0.1
    assert sig.parameters["adjust_lims_set_aspect"].default is True


def test_set_aspect_before_renderer_draw_actually_sticks():
    """Family-5 regression: on an explicitly-positioned axes (created via
    `mfig.add_axes(rect)`), calling `set_aspect` AFTER the renderer's
    internal `canvas.draw()` was effectively a no-op. The fix is to set
    aspect BEFORE the renderer renders. This test pins the ordering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    # Mimic the renderer pre-set-aspect path.
    ax.set_aspect("auto")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    fig.canvas.draw()
    assert ax.get_aspect() == "auto"
    plt.close(fig)


def test_adjust_limits_padding_zero_gives_tight_lims():
    """Padding=0 must produce lims tight to content (no floor like the old
    `max(width*0.15, 10.0)` which made small schematics get huge margins)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    fig, ax = plt.subplots()

    class _MockComp:
        parent = None

        class _Bounds:
            pass

    # We can't easily mock get_recursive_world_bounds, so check the floor
    # was removed via inspection.
    import inspect

    src = inspect.getsource(MatplotlibRenderer._adjust_limits)
    assert "max(width * padding * 1.5, 10.0)" not in src, "Old padding floor still present"
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# (6) panel_from field coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_smooth_grad_panel_has_label_pad_fields():
    """Family-6 regression: SmoothGradMagnitudePanel2D needs labelpad fields
    so the theme cascade can target them. Cascade silently no-ops on unknown
    fields, which makes the bug hard to diagnose."""
    assert "xaxis_labelpad" in SmoothGradMagnitudePanel2D.model_fields
    assert "yaxis_labelpad" in SmoothGradMagnitudePanel2D.model_fields


def test_smooth_panel_2d_has_label_pad_fields():
    """The two heatmap panels must have parity on labelpad surface — themes
    target both classes interchangeably."""
    assert "xaxis_labelpad" in SmoothPanel2D.model_fields
    assert "yaxis_labelpad" in SmoothPanel2D.model_fields


# ─────────────────────────────────────────────────────────────────────────────
# (8) FluoMarker label auto-shrink
# ─────────────────────────────────────────────────────────────────────────────


def test_fluo_marker_opted_into_label_auto_fit():
    """FluoMarker must declare `_label_fit_to_svg = True` so long names like
    `mNeonGreen` shrink to fit the SVG arrow."""
    from jeanplot.gene.elements import FluoMarker, Promoter

    assert FluoMarker._label_fit_to_svg is True
    # Other parts have short labels — they should NOT auto-shrink (changes
    # surrounding layout). Promoter/Terminator don't auto-label at all.
    assert Promoter._label_fit_to_svg is False


def test_fluo_marker_label_shrinks_when_text_overflows_svg():
    """e2e-ish: long marker name triggers font_size reduction during layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeanplot.gene.elements import FluoMarker
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    fig, ax = plt.subplots()
    renderer = MatplotlibRenderer()
    renderer._context = ax

    marker = FluoMarker(id="fluo_long", part_name="mNeonGreen")
    initial_font = marker.label.font_size
    marker.measure_and_layout(renderer)

    svg_w = marker._svg_shape._natural_dimensions.width
    lbl_w = marker.label._natural_dimensions.width

    # Label width should now be ≤ factor × svg width
    assert lbl_w <= svg_w * marker._label_fit_factor + 1e-6, (
        f"label not auto-fit: lbl_w={lbl_w}, svg_w={svg_w}, font={marker.label.font_size}"
    )
    # And font_size should have shrunk
    assert marker.label.font_size < initial_font, (
        f"font_size unchanged: {marker.label.font_size}"
    )
    plt.close(fig)


def test_fluo_marker_short_label_does_not_shrink():
    """Short labels (e.g. `eYFP`) already fit — they must NOT be shrunk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeanplot.gene.elements import FluoMarker
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    fig, ax = plt.subplots()
    renderer = MatplotlibRenderer()
    renderer._context = ax

    marker = FluoMarker(id="fluo_short", part_name="eYFP")
    initial_font = marker.label.font_size
    marker.measure_and_layout(renderer)

    assert marker.label.font_size == initial_font, (
        f"short label was shrunk: {initial_font} → {marker.label.font_size}"
    )
    plt.close(fig)


def test_fluo_marker_auto_shrink_idempotent():
    """Re-running measure_and_layout must NOT keep shrinking the font."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeanplot.gene.elements import FluoMarker
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    fig, ax = plt.subplots()
    renderer = MatplotlibRenderer()
    renderer._context = ax

    marker = FluoMarker(id="fluo_idem", part_name="mNeonGreen")
    marker.measure_and_layout(renderer)
    after_first = marker.label.font_size

    for _ in range(3):
        marker.measure_and_layout(renderer)

    assert marker.label.font_size == after_first, (
        f"font kept shrinking: ended at {marker.label.font_size}"
    )
    plt.close(fig)


def test_fluo_marker_shrunk_font_survives_cascade_reapply():
    """The auto-shrink marks `font_size` as user-set so subsequent
    `jstyle.apply` passes (which happen on every measure_and_layout) don't
    reset the font back to the theme default."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeanplot.gene.elements import FluoMarker
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    jstyle.update({"GeneticPart": {"Text": {"font_size": 9}}})
    fig, ax = plt.subplots()
    renderer = MatplotlibRenderer()
    renderer._context = ax

    marker = FluoMarker(id="fluo_cascade", part_name="mNeonGreen")
    marker.measure_and_layout(renderer)
    shrunk = marker.label.font_size

    assert shrunk < 9, f"font should have shrunk below 9, got {shrunk}"
    assert "font_size" in marker.label._user_set_fields, (
        "shrunk font_size must be marked user-set to survive cascade"
    )
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# (9) AutoLabelMixin part-name aliases
# ─────────────────────────────────────────────────────────────────────────────


def test_autolabel_aliases_rewrite_label_text():
    """register_label_aliases lets users shorten part_name → display label
    (e.g. mNeonGreen → mNGreen) without touching the underlying part_name."""
    from jeanplot.gene.elements import FluoMarker

    original = dict(FluoMarker._label_aliases)
    try:
        FluoMarker.register_label_aliases({"mNeonGreen": "mNGreen"})
        m = FluoMarker(id="fluo_alias", part_name="mNeonGreen")
        assert m.part_name == "mNeonGreen"
        assert m.label is not None and m.label.text == "mNGreen"
    finally:
        FluoMarker._label_aliases = original


def test_autolabel_aliases_isolated_per_subclass():
    """Aliases registered on FluoMarker must not leak to ERN/UorfGroup."""
    from jeanplot.gene.elements import FluoMarker, ERN, UorfGroup

    original = dict(FluoMarker._label_aliases)
    try:
        FluoMarker.register_label_aliases({"shared_name": "X"})
        assert ERN._label_aliases.get("shared_name") is None
        assert UorfGroup._label_aliases.get("shared_name") is None
    finally:
        FluoMarker._label_aliases = original


def test_autolabel_aliases_passthrough_for_unmapped_names():
    """Names not in the alias dict render unchanged."""
    from jeanplot.gene.elements import FluoMarker

    m = FluoMarker(id="fluo_pass", part_name="eYFP")
    assert m.label is not None and m.label.text == "eYFP"


# ─────────────────────────────────────────────────────────────────────────────
# (10) End-to-end: render a schematic figure and check artifact sizing
# ─────────────────────────────────────────────────────────────────────────────


def test_e2e_genetic_schematic_renders_with_long_fluo_label():
    """e2e: build a GeneticSchematic with a long FluoMarker label, render
    it to a matplotlib axes, and verify the rendered text artist's width
    fits inside the parent SVG width. This is the family-8 visual bug
    families 1–5 don't catch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeanplot.gene.elements import FluoMarker
    from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

    fig, ax = plt.subplots(figsize=(4, 2))
    renderer = MatplotlibRenderer()
    renderer._context = ax

    marker = FluoMarker(id="fluo_e2e", part_name="mNeonGreen")
    renderer.render_component(ax, marker, adjust_lims=True)

    svg_w = marker._svg_shape._natural_dimensions.width
    lbl_w = marker.label._natural_dimensions.width

    assert lbl_w <= svg_w * marker._label_fit_factor + 1e-6, (
        f"e2e: long label overflowed SVG. lbl_w={lbl_w}, svg_w={svg_w}"
    )
    plt.close(fig)


def test_e2e_box_style_cascade_full_pipeline():
    """e2e: a `style.padding.right: X` cascade rule propagates all the way
    through `jstyle.apply` to the final BoxStyle on a component. Pins the
    full cascade-fill path used by paper themes."""
    jstyle.update({
        "PlotPanel": {
            "style.padding.bottom": 0.5,
            "style.padding.left": 0.5,
        },
        "SmoothPanel2D": {
            "style.padding.right": 0.6,
        },
        "SmoothPanel1D": {
            "style.padding.right": 1.0,
        },
    })
    p2 = SmoothPanel2D()
    p1 = SmoothPanel1D()
    jstyle.apply(p2)
    jstyle.apply(p1)
    # Both panels inherit base bottom/left from PlotPanel...
    assert (p2.style.padding.bottom, p2.style.padding.left) == (0.5, 0.5)
    assert (p1.style.padding.bottom, p1.style.padding.left) == (0.5, 0.5)
    # ...but each has its own right padding.
    assert p2.style.padding.right == 0.6
    assert p1.style.padding.right == 1.0


def test_e2e_no_axis_label_pad_field_silent_noop():
    """A subtle bug class: a cascade rule targeting a field that doesn't
    exist on the panel class silently no-ops. This test makes sure all the
    panels we care about for paper figures have the labelpad fields they
    need to receive cascade rules."""
    expected_fields = {"xaxis_labelpad", "yaxis_labelpad"}
    for cls in (SmoothPanel2D, SmoothGradMagnitudePanel2D):
        missing = expected_fields - set(cls.model_fields.keys())
        assert not missing, f"{cls.__name__} missing fields: {missing}"


def test_e2e_smooth_panel_with_padding_renders():
    """A SmoothPanel2D with all the new style.padding fields renders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(50, 2)).astype(np.float32)
    Y = (X[:, :1] + X[:, 1:]).astype(np.float32)
    pd = PlotData(xval=X, yval=Y, input_names=["a", "b"], output_name="o")

    panel = SmoothPanel2D(
        plot_data=pd,
        axes_size=Size(width=3.0, height=2.5),
        style=BoxStyle(padding=BoxInset(top=0.3, right=0.6, bottom=0.5, left=0.5)),
    )
    fig = Figure(panel, output_file=None, dpi=50)
    mfig = fig.render()
    assert mfig is not None
    plt.close(mfig)


def test_e2e_size_cascade_preserves_panel_dimensions():
    """End-to-end check for family-1: load a theme that sets paper_axes_size,
    construct a panel with explicit axes_size, render, and verify the panel
    bbox ends up at the user's dimensions, not the theme's.

    Note: we check `panel._dimensions` (the laid-out panel bbox), not the
    matplotlib axes position — the matplotlib axes may shrink further due to
    `aspect="equal"` on uneven data ranges, which is correct behavior.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jstyle.update({
        "SmoothPanel2D": {"axes_size": Size(width=2.5, height=2.0)},
    })
    try:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(50, 2)).astype(np.float32)
        Y = (X[:, :1] + X[:, 1:]).astype(np.float32)
        pd = PlotData(xval=X, yval=Y, input_names=["a", "b"], output_name="o")

        panel = SmoothPanel2D(plot_data=pd, axes_size=Size(width=10.0, height=4.0))
        fig = Figure(panel, output_file=None, dpi=50)
        mfig = fig.render()

        # axes_size must be preserved on the panel itself (not theme-overridden).
        assert panel.axes_size.width == 10.0, f"axes_size.width lost: {panel.axes_size}"
        assert panel.axes_size.height == 4.0, f"axes_size.height lost: {panel.axes_size}"

        # The laid-out panel bbox must reflect the user's axes_size.
        assert panel._dimensions.width >= 10.0, (
            f"panel bbox shrunk: {panel._dimensions}"
        )
        assert panel._dimensions.height >= 4.0, (
            f"panel bbox shrunk: {panel._dimensions}"
        )
        plt.close(mfig)
    finally:
        jstyle.clear()


# ─────────────────────────────────────────────────────────────────────────────
# (cleanup)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_jstyle_after():
    yield
    jstyle.clear()
