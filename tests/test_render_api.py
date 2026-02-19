"""Tests for top-level render API."""

import matplotlib.pyplot as plt

from jeanplot import Container, Size
from jeanplot.render import render


def test_render_svg_returns_string_without_output():
    comp = Container(id="box", min_dimensions=Size(80, 40))
    svg = render(comp, backend="svg")
    assert isinstance(svg, str)
    assert "<svg" in svg


def test_render_matplotlib_returns_axes_when_context_provided():
    comp = Container(id="box", min_dimensions=Size(80, 40))
    fig, ax = plt.subplots()
    try:
        out = render(comp, backend="matplotlib", context=ax)
        assert out is ax
    finally:
        plt.close(fig)
