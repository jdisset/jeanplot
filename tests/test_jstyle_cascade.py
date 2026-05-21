"""Cascade-based jstyle behaviors: basic apply, specificity, descendant
nesting, and live-scope `${component.X}` resolution."""

import dracon as dr
from pydantic import Field

from jeanplot import (
    Component,
    Container,
    Text,
    BoxStyle,
    jstyle,
    DEFAULT_TYPES,
    make_context_from_types,
)


class CascadeComp(Component):
    color: str = "black"
    size: int = 10
    name: str = ""
    style: BoxStyle = Field(default_factory=BoxStyle)


class SpecialContainer(Container):
    pass


# ── basic ────────────────────────────────────────────────────────────────────


def test_single_rule_applies_to_matching_component():
    jstyle.update({"CascadeComp": {"color": "red"}})
    c = CascadeComp(id="c1")
    jstyle.apply(c)
    assert c.color == "red"


def test_single_rule_leaves_non_matching_alone():
    jstyle.update({"CascadeComp": {"color": "red"}})
    t = Text(id="t1", text="hi")
    jstyle.apply(t)
    assert t.color != "red"


# ── specificity ──────────────────────────────────────────────────────────────


def test_class_attr_beats_type():
    jstyle.update({
        "CascadeComp": {"color": "red"},
        "[style_class=hot]": {"color": "orange"},
    })
    a = CascadeComp(id="a")
    b = CascadeComp(id="b", style_class=["hot"])
    root = Container(children=[a, b])
    jstyle.apply(root)
    assert a.color == "red"
    assert b.color == "orange"


def test_id_beats_class_attr():
    jstyle.update({
        "[style_class=hot]": {"color": "orange"},
        "[id=star]": {"color": "purple"},
    })
    a = CascadeComp(id="a", style_class=["hot"])
    b = CascadeComp(id="star", style_class=["hot"])
    root = Container(children=[a, b])
    jstyle.apply(root)
    assert a.color == "orange"
    assert b.color == "purple"


# ── descendant ───────────────────────────────────────────────────────────────


def test_descendant_only_matches_inside_ancestor():
    jstyle.update({"Container": {"Text": {"color": "red"}}})
    outside = Text(id="out", text="x")
    inside = Text(id="in", text="y")
    c = Container(id="c", children=[inside])
    jstyle.apply(outside)
    jstyle.apply(c)
    assert outside.color != "red"
    assert inside.color == "red"


def test_descendant_deep_ancestor_chain():
    jstyle.update({"SpecialContainer": {"Text": {"color": "green"}}})
    txt = Text(id="t", text="x")
    inner = Container(id="inner", children=[txt])
    outer = SpecialContainer(id="outer", children=[inner])
    jstyle.apply(outer)
    assert txt.color == "green"


# ── live-scope ───────────────────────────────────────────────────────────────


def test_live_scope_resolves_against_component_attribute(tmp_path):
    yaml_text = (
        "rules: !cascade:jstyle\n"
        "  CascadeComp:\n"
        "    color: ${component.name + '-tag'}\n"
    )
    cfg_path = tmp_path / "live_theme.yaml"
    cfg_path.write_text(yaml_text)

    cfg = dr.load(
        f"file:{cfg_path}",
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES + [CascadeComp]),
    )
    dr.resolve_all_lazy(cfg, except_for={"component"})

    jstyle.clear()
    jstyle.update(cfg["rules"])

    a = CascadeComp(id="a", name="alpha")
    b = CascadeComp(id="b", name="beta")
    jstyle.apply(a)
    jstyle.apply(b)
    assert a.color == "alpha-tag"
    assert b.color == "beta-tag"
