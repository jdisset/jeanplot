"""Layout-string DSL: ``layout="row gap=1.0 align=stretch"`` parses to a
``LayoutConstraints`` equivalent to the long-form mapping / typed instance."""

import pytest
import dracon

from jeanplot import Container, LayoutConstraints, make_plot_context


def test_typed_instance_still_works():
    c = Container(layout=LayoutConstraints(direction="row", gap=2.0))
    assert c.layout.direction == "row"
    assert c.layout.gap == 2.0


def test_mapping_input_still_works():
    c = Container(layout={"direction": "row", "gap": 1.5})
    assert c.layout.direction == "row"
    assert c.layout.gap == 1.5


def test_string_basic():
    c = Container(layout="row gap=1.0")
    assert c.layout.direction == "row"
    assert c.layout.gap == 1.0


def test_string_with_align_alias():
    c = Container(layout="col align=stretch")
    assert c.layout.direction == "column"
    assert c.layout.align_items == "stretch"


def test_string_with_justify_alias():
    c = Container(layout="row justify=center gap=2")
    assert c.layout.direction == "row"
    assert c.layout.justify_content == "center"
    assert c.layout.gap == 2


def test_string_full_round_trip_matches_long_form():
    a = Container(layout="row gap=1.5 align=stretch justify=center")
    b = Container(
        layout=LayoutConstraints(
            direction="row", gap=1.5, align_items="stretch", justify_content="center"
        )
    )
    assert a.layout.model_dump() == b.layout.model_dump()


def test_invalid_direction_raises():
    with pytest.raises(Exception, match="row|column"):
        Container(layout="diagonal gap=1")


def test_invalid_kv_raises():
    with pytest.raises(Exception, match="key=value"):
        Container(layout="row gap")


def test_yaml_layout_string():
    src = """
!Container
layout: "row gap=1.0 align=stretch"
"""
    c = dracon.loads(src, context=make_plot_context())
    assert isinstance(c, Container)
    assert c.layout.direction == "row"
    assert c.layout.gap == 1.0
    assert c.layout.align_items == "stretch"
