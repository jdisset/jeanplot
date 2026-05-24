"""Cascade-fill semantics: theme rules are DEFAULTS, explicit user values win.

This is the load-bearing invariant of refactor 01. The old clobber-only
behavior is gone; without this test the next person to touch
``style_engine.py`` will silently restore it.
"""

import dracon as dr
from pydantic import Field

from jeanplot import (
    Component,
    Container,
    BoxStyle,
    DEFAULT_TYPES,
    make_context_from_types,
    jstyle,
)
from jeanplot.core.models import Offset


class FillComp(Component):
    color: str = "black"
    size: int = 10
    meta: dict = Field(default_factory=dict)
    style: BoxStyle = Field(default_factory=BoxStyle)
    # non-zero default like Connection.start_offset: a partial theme value must replace it
    anchor: Offset = Field(default_factory=lambda: Offset(reference_relative=(0.5, 0.5)))


def test_explicit_value_wins_over_theme():
    jstyle.update({"FillComp": {"color": "red"}})
    c = FillComp(id="c", color="purple")
    jstyle.apply(c)
    assert c.color == "purple"


def test_theme_fills_missing_field():
    jstyle.update({"FillComp": {"color": "red"}})
    c = FillComp(id="c")
    jstyle.apply(c)
    assert c.color == "red"


def test_nested_dict_merges_per_key():
    jstyle.update({"FillComp": {"meta": {"cmap": "viridis", "bad_color": "#fff"}}})
    c = FillComp(id="c", meta={"cmap": "magma"})
    jstyle.apply(c)
    assert c.meta == {"cmap": "magma", "bad_color": "#fff"}


def test_nested_model_per_field_user_wins():
    jstyle.update(
        {
            "FillComp": {
                "style": {"background_color": "green", "border_width": 4.0},
            }
        }
    )
    c = FillComp(id="c", style=BoxStyle(border_width=1.0))
    jstyle.apply(c)
    assert c.style.border_width == 1.0
    assert c.style.background_color in ("green", "#008000ff")


def test_typed_model_fill_replaces_field_default():
    # complete same-type value replaces; field default_factory must not leak through
    jstyle.update({"FillComp": {"anchor": Offset(relative=(1.0, 0.5))}})
    c = FillComp(id="c")
    jstyle.apply(c)
    assert c.anchor.relative == (1.0, 0.5)
    assert c.anchor.reference_relative == (0.0, 0.0)


def test_typed_model_user_set_deep_fills():
    # user set the field -> keep their sub-fields, fill only what they omitted
    jstyle.update({"FillComp": {"anchor": Offset(reference_relative=(0.1, 0.2))}})
    c = FillComp(id="c", anchor=Offset(relative=(0.3, 0.0)))
    jstyle.apply(c)
    assert c.anchor.relative == (0.3, 0.0)
    assert c.anchor.reference_relative == (0.1, 0.2)


def test_typed_model_clobber_replaces(tmp_path):
    yaml_text = "rules: !cascade:jstyle\n  FillComp:\n    anchor: !Offset { relative: [1, 0.5] }\n"
    p = tmp_path / "clobber_model.yaml"
    p.write_text(yaml_text)
    cfg = dr.load(
        f"file:{p}",
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES + [FillComp]),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    jstyle.clear()
    jstyle.update(cfg["rules"])
    try:
        c = FillComp(id="c", anchor=Offset(reference_relative=(0.9, 0.9)))
        jstyle.apply(c)
        assert c.anchor.relative == (1.0, 0.5)
        assert c.anchor.reference_relative == (0.0, 0.0)
    finally:
        jstyle.clear()


def test_descendant_selectors_still_apply_under_fill():
    jstyle.update({"Container": {"FillComp": {"color": "lime"}}})
    inner = FillComp(id="inner")
    outer = Container(id="o", children=[inner])
    jstyle.apply(outer)
    assert inner.color == "lime"


def test_reapplying_jstyle_does_not_escalate():
    jstyle.update({"FillComp": {"color": "red"}})
    c = FillComp(id="c", color="purple")
    for _ in range(3):
        jstyle.apply(c)
    assert c.color == "purple"


def test_reapplying_does_not_pollute_user_set_fields():
    jstyle.update({"FillComp": {"color": "red", "size": 99}})
    c = FillComp(id="c")
    # First pass: theme fills color + size.
    jstyle.apply(c)
    assert c.color == "red"
    assert c.size == 99
    # Now flip the theme; the first-pass writes must NOT have become user-set.
    jstyle.update({"FillComp": {"color": "blue", "size": 1}})
    jstyle.apply(c)
    assert c.color == "blue"
    assert c.size == 1


def test_user_set_fields_excludes_defaults():
    c = FillComp(id="c", color="purple")
    assert "color" in c._user_set_fields
    assert "size" not in c._user_set_fields


def test_bare_list_path_records_children_as_user_set():
    inner = FillComp(id="i")
    c = Container.model_validate([inner])
    assert "children" in c._user_set_fields


def test_positional_path_records_children_as_user_set():
    inner = FillComp(id="i")
    c = Container(inner)
    assert "children" in c._user_set_fields


def test_jstyle_clobber_strategy_still_clobbers(tmp_path):
    yaml_text = "rules: !cascade:jstyle\n  FillComp:\n    color: red\n"
    p = tmp_path / "clobber.yaml"
    p.write_text(yaml_text)
    cfg = dr.load(
        f"file:{p}",
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES + [FillComp]),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    jstyle.clear()
    jstyle.update(cfg["rules"])
    try:
        c = FillComp(id="c", color="purple")
        jstyle.apply(c)
        assert c.color == "red"
    finally:
        jstyle.clear()


def test_jstyle_fill_yaml_strategy_fills(tmp_path):
    yaml_text = "rules: !cascade:jstyle_fill\n  FillComp:\n    color: red\n"
    p = tmp_path / "fill.yaml"
    p.write_text(yaml_text)
    cfg = dr.load(
        f"file:{p}",
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES + [FillComp]),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})
    jstyle.clear()
    jstyle.update(cfg["rules"])
    try:
        c_set = FillComp(id="a", color="purple")
        c_default = FillComp(id="b")
        jstyle.apply(c_set)
        jstyle.apply(c_default)
        assert c_set.color == "purple"
        assert c_default.color == "red"
    finally:
        jstyle.clear()
