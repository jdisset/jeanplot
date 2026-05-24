"""Regression tests for cascade specificity tiebreakers.

Covers two bug classes encountered when authoring the default jeanplot theme:

1. **Auto-created child components must not pre-set visual fields.** When a
   parent (e.g. `AutoLabelMixin`) hardcodes `font_size`/`offset`/etc. as Text
   kwargs, those fields land in `Text._user_set_fields` and `jstyle_fill`
   refuses to override them. Theme rules become silently ineffective.

2. **Descendant selectors should prefer the snuggest ancestor match.** When
   two selectors `A Text` and `B Text` tie on CSS specificity but `A` is the
   element's immediate parent and `B` is its grandparent, `A Text` should win
   regardless of source order. CSS doesn't distinguish, but for jeanplot's
   scene-graph use case "closer ancestor wins" matches authorial intent."""

from pydantic import Field

from jeanplot import Component, Container, Text, BoxStyle, jstyle
from jeanplot.core.style_selector import Selector
from jeanplot.gene.elements import ERN, FluoMarker, UorfGroup


# ─────────────────────────────────────────────────────────────────────────────
# (1) AutoLabelMixin must not pre-set visual fields on auto-labels
# ─────────────────────────────────────────────────────────────────────────────


def _auto_label_for(part):
    """The Text child auto-created by AutoLabelMixin."""
    return part.label


def test_autolabel_does_not_preset_font_size_on_fluo_marker():
    fm = FluoMarker(id="fm", part_name="eYFP")
    label = _auto_label_for(fm)
    assert "font_size" not in label.model_fields_set, (
        "font_size in model_fields_set blocks jstyle_fill from overriding"
    )


def test_autolabel_does_not_preset_font_size_on_ern():
    e = ERN(id="ern", part_name="CasE")
    label = _auto_label_for(e)
    assert "font_size" not in label.model_fields_set


def test_autolabel_does_not_preset_font_size_or_offset_on_uorf():
    u = UorfGroup(id="u", part_name="1xuORF")
    label = _auto_label_for(u)
    assert "font_size" not in label.model_fields_set
    assert "offset" not in label.model_fields_set


def test_theme_font_size_propagates_to_autolabel():
    """Regression: with the AutoLabelMixin fix, a `GeneticPart Text` theme rule
    must actually change the auto-label's rendered font_size."""
    jstyle.update({"GeneticPart": {"Text": {"font_size": 42}}})
    try:
        fm = FluoMarker(id="fm_theme", part_name="eYFP")
        jstyle.apply(fm)
        assert fm.label.font_size == 42
    finally:
        jstyle.clear()


# ─────────────────────────────────────────────────────────────────────────────
# (2) Ancestor-skip distance as a primary specificity tiebreaker
# ─────────────────────────────────────────────────────────────────────────────


class Outer(Container):
    pass


class Middle(Container):
    pass


class Inner(Container):
    pass


def _wire_parents(root):
    for child in getattr(root, "children", None) or []:
        child.parent = root
        _wire_parents(child)


def _build_outer_middle_inner_text() -> tuple[Outer, Text]:
    """Tree: Outer > Middle > Inner > Text. Returns (root, leaf) with parents wired
    (Container's constructor accepts `children=...` but doesn't backlink `.parent`;
    `jstyle.apply` normally fixes that on walk, but pure-selector probes need it eagerly)."""
    txt = Text(id="leaf", text="x")
    inner = Inner(id="inner", children=[txt])
    middle = Middle(id="middle", children=[inner])
    outer = Outer(id="outer", children=[middle])
    _wire_parents(outer)
    return outer, txt


def test_get_inexactness_immediate_parent_is_zero():
    outer, leaf = _build_outer_middle_inner_text()
    skip, mro = Selector("Inner Text").get_inexactness(leaf)
    assert (skip, mro) == (0, 0), "immediate-parent ancestor match should be (0, 0)"


def test_get_inexactness_grandparent_has_skip_one():
    outer, leaf = _build_outer_middle_inner_text()
    skip, mro = Selector("Middle Text").get_inexactness(leaf)
    assert skip == 1, "grandparent match must show skip=1 (Inner was skipped)"


def test_get_inexactness_two_skips_for_great_grandparent():
    outer, leaf = _build_outer_middle_inner_text()
    skip, mro = Selector("Outer Text").get_inexactness(leaf)
    assert skip == 2


def test_immediate_parent_selector_beats_grandparent_via_source_order():
    """`Inner Text` and `Middle Text` tie on CSS specificity (0,0,2). With the
    ancestor-skip tiebreaker, `Inner Text` (skip=0) must beat `Middle Text`
    (skip=1) even when Middle is declared LATER in the source — proving the
    tiebreaker isn't just relying on source order."""
    jstyle.update({
        "Inner": {"Text": {"font_size": 11}},
        "Middle": {"Text": {"font_size": 22}},  # later in source order
    })
    try:
        outer, leaf = _build_outer_middle_inner_text()
        jstyle.apply(outer)  # apply from Outer down
        assert leaf.font_size == 11, "immediate-parent rule must win over grandparent rule"
    finally:
        jstyle.clear()


def test_grandparent_selector_beats_great_grandparent_via_source_order():
    """Same shape, deeper: `Middle Text` (skip=1) beats `Outer Text` (skip=2)."""
    jstyle.update({
        "Middle": {"Text": {"font_size": 33}},
        "Outer":  {"Text": {"font_size": 44}},  # later in source order
    })
    try:
        outer, leaf = _build_outer_middle_inner_text()
        jstyle.apply(outer)
        assert leaf.font_size == 33
    finally:
        jstyle.clear()


def test_immediate_parent_beats_grandparent_for_disjoint_properties():
    """If `Inner Text` and `Middle Text` each set DIFFERENT props, both should
    apply — but where they overlap, Inner Text wins."""
    jstyle.update({
        "Middle": {"Text": {"font_size": 50, "color": "red"}},
        "Inner":  {"Text": {"font_size": 60}},
    })
    try:
        outer, leaf = _build_outer_middle_inner_text()
        jstyle.apply(outer)
        assert leaf.font_size == 60, "Inner Text wins for shared 'font_size'"
        assert leaf.color == "red", "Middle Text's unique 'color' still merges in"
    finally:
        jstyle.clear()


def test_skip_dominates_mro_when_both_differ():
    """`Container Text` (skip=0, mro=1 — Inner inherits Container) vs
    `Middle Text` (skip=1, mro=0 — Middle is exact class). Skip dominates,
    so the snug-but-inherited rule wins over the distant-but-exact one."""
    jstyle.update({
        "Container": {"Text": {"font_size": 70}},  # snug + inherited
        "Middle":    {"Text": {"font_size": 80}},  # distant + exact
    })
    try:
        outer, leaf = _build_outer_middle_inner_text()
        jstyle.apply(outer)
        assert leaf.font_size == 70, "snug-ancestor rule wins even when its match is inherited"
    finally:
        jstyle.clear()


# ─────────────────────────────────────────────────────────────────────────────
# (3) End-to-end: real default-theme regression for FluoMarker label
# ─────────────────────────────────────────────────────────────────────────────


def test_fluomarker_label_inside_tu_picks_up_geneticpart_text_rule():
    """The original bug: a FluoMarker label inside a TranscriptionUnit was
    styled by `TranscriptionUnit Text` (distant ancestor) instead of by
    `FluoMarker Text` / `GeneticPart Text` (snug ancestors). Verify with a
    minimal real-shaped theme that the snug rule wins."""
    from jeanplot.gene.elements import TranscriptionUnit

    jstyle.update({
        "TranscriptionUnit": {"Text": {"font_size": 5, "color": "#999"}},  # distant
        "GeneticPart":       {"Text": {"font_size": 30}},                  # snug-inherited
        "FluoMarker":        {"Text": {"color": "#9b6600"}},               # snug-exact
    })
    try:
        fm = FluoMarker(id="fm_in_tu", part_name="eYFP")
        tu = TranscriptionUnit(id="tu", children=[fm])
        jstyle.apply(tu)
        assert fm.label.font_size == 30, "GeneticPart Text (snug-inherited) beats TU Text (distant)"
        assert fm.label.color == "#9b6600", "FluoMarker Text (snug-exact) wins for color"
    finally:
        jstyle.clear()


# ─────────────────────────────────────────────────────────────────────────────
# (4) Library-set defaults via `with_defaults` must remain theme-overridable
# ─────────────────────────────────────────────────────────────────────────────


def test_with_defaults_does_not_mark_fields_user_set():
    t = Text(id="t", text="x").with_defaults(font_size=99, color="#abcdef")
    assert "font_size" not in t._user_set_fields
    assert "color" not in t._user_set_fields
    assert t.font_size == 99 and t.color == "#abcdef"


def test_with_defaults_preserves_real_user_input():
    t = Text(id="t", text="x", font_size=42).with_defaults(font_size=99, color="#abcdef")
    assert "font_size" in t._user_set_fields
    assert t.font_size == 42
    assert t.color == "#abcdef"


def test_theme_overrides_library_default():
    jstyle.update({"Text": {"font_size": 7}})
    try:
        t = Text(id="t", text="x").with_defaults(font_size=99)
        jstyle.apply(t)
        assert t.font_size == 7
    finally:
        jstyle.clear()


def test_theme_loses_against_real_user_input():
    jstyle.update({"Text": {"font_size": 7}})
    try:
        t = Text(id="t", text="x", font_size=42)
        jstyle.apply(t)
        assert t.font_size == 42
    finally:
        jstyle.clear()


def test_fill_merges_basemodel_into_basemodel():
    """Engine fix: when current_val is a BaseModel and the theme value is ALSO
    a BaseModel (e.g. `!LineEndFlat ...`), fill semantics must merge field-by-
    field instead of bailing."""
    from jeanplot.core.connector import Connection
    from jeanplot.core.svg import LineEndFlat

    conn = Connection(
        id="c",
        start_component="//a",
        end_component="//b",
        end_cap=LineEndFlat(stroke_width=1.5, length=8.0),
    )
    jstyle.update({"Connection": {"end_cap": LineEndFlat(stroke_color="#999")}})
    try:
        jstyle.apply(conn)
        assert conn.end_cap.stroke_color == "#999"
        assert conn.end_cap.stroke_width == 1.5
        assert conn.end_cap.length == 8.0
    finally:
        jstyle.clear()
