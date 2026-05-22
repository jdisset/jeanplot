"""`!Container [a, b, c]` is sugar for `!Container { children: [a, b, c] }`.

The same shorthand inherits to every Container subclass (Figure, PlotPanel, ...).
"""

import dracon

from jeanplot import Container, Figure, make_plot_context
from jeanplot.core.text import Text


def test_container_model_validate_from_list():
    c = Container.model_validate([Text(text="a"), Text(text="b")])
    assert [t.text for t in c.children] == ["a", "b"]


def test_container_model_validate_from_dict_still_works():
    c = Container.model_validate({"children": [Text(text="x")]})
    assert [t.text for t in c.children] == ["x"]


def test_figure_inherits_bare_list_shorthand():
    f = Figure.model_validate([Text(text="hi")])
    assert isinstance(f, Figure)
    assert [t.text for t in f.children] == ["hi"]


def test_yaml_container_tag_with_bare_sequence():
    src = """
!Container
- !Text { text: a }
- !Text { text: b }
"""
    c = dracon.loads(src, context=make_plot_context())
    assert isinstance(c, Container)
    assert [t.text for t in c.children] == ["a", "b"]


def test_yaml_figure_tag_with_bare_sequence():
    src = """
!Figure
- !Text { text: hello }
"""
    f = dracon.loads(src, context=make_plot_context())
    assert isinstance(f, Figure)
    assert len(f.children) == 1
