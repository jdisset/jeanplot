"""Positional ``*children`` on ``Container`` / ``Figure`` (and any subclass).

``Container(a, b)`` works the same as ``Container(children=[a, b])`` and the
same as ``!Container [a, b]`` in YAML. Mutually exclusive with ``children=``.
"""

import pytest

from jeanplot import Container, Figure
from jeanplot.core.text import Text


def test_positional_children_two():
    c = Container(Text(text="a"), Text(text="b"))
    assert [t.text for t in c.children] == ["a", "b"]


def test_positional_children_one_figure():
    f = Figure(Text(text="x"))
    assert isinstance(f, Figure)
    assert [t.text for t in f.children] == ["x"]


def test_positional_children_with_kwargs_still_works():
    f = Figure(Text(text="x"), dpi=72)
    assert f.dpi == 72
    assert [t.text for t in f.children] == ["x"]


def test_positional_and_children_kwarg_are_exclusive():
    with pytest.raises(TypeError, match="mutually exclusive"):
        Container(Text(text="a"), children=[Text(text="b")])


def test_positional_path_records_children_user_set():
    c = Container(Text(text="a"))
    assert "children" in c._user_set_fields
