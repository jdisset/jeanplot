"""bugs/_archive/2026-05-21-straightcurve-start-vector-spam.md"""

import logging
import re
from pathlib import Path

import pytest

import jeanplot
from jeanplot import Connection, Container, Size, jstyle
from jeanplot.core.curve import OrthogonalCurve, SimpleBezierCurve


class ERNNode(Container):  # selector matches by class name; biocomp-free stand-in
    pass


def _scene(style_class):
    a = Container(id="a", min_dimensions=Size(40, 20))
    b = Container(id="b", min_dimensions=Size(40, 20))
    conn = Connection(style_class=[style_class], start_component="a", end_component="b")
    return ERNNode(id="ern", min_dimensions=Size(200, 100), children=[a, b, conn]), conn


@pytest.fixture
def default_theme():
    jeanplot.load_default_theme(force=True)
    yield
    jstyle.clear()


@pytest.mark.parametrize("style_class, curve_cls", [
    ("txconn", SimpleBezierCurve),
    ("tlconn", OrthogonalCurve),
])
def test_default_theme_apply_is_clean(default_theme, caplog, style_class, curve_cls):
    ern, conn = _scene(style_class)
    with caplog.at_level(logging.WARNING, logger="jeanplot.core.style_engine"):
        jstyle.apply(ern)
    noise = [r for r in caplog.records
             if r.name == "jeanplot.core.style_engine" and r.levelno >= logging.WARNING]
    assert not noise, "\n".join(f"{r.levelname}: {r.getMessage()}" for r in noise)
    assert isinstance(conn.curve_type, curve_cls)


def test_cascade_hands_out_independent_instances(default_theme):
    # cascade rule is a spec, not a singleton -- each matched component
    # owns its polymorphic field state (mutable PrivateAttrs etc.)
    scenes = [_scene("tlconn") for _ in range(2)]
    for ern, _ in scenes:
        jstyle.apply(ern)
    c1, c2 = scenes[0][1], scenes[1][1]
    assert c1.curve_type is not c2.curve_type
    assert c1.end_cap is not c2.end_cap
    c1.curve_type.get_path((0, 0), (10, 5))
    c2.curve_type.get_path((100, 100), (200, 300))
    assert c1.curve_type._cached_path_points != c2.curve_type._cached_path_points


# polymorphic fields on Connection: typed Union or `| None`. Flat-path
# (`curve_type.x: ...`) silently assumes a subclass; whole-object only.
_POLY_FIELDS = ("curve_type", "end_cap", "start_cap")
_FLAT_RE = re.compile(rf"^\s*({'|'.join(_POLY_FIELDS)})\.(\w+)\s*:", re.MULTILINE)


def _themes():
    return sorted((Path(jeanplot.__file__).parent / "resources" / "themes").glob("*.yaml"))


@pytest.mark.parametrize("theme", _themes(), ids=lambda p: p.name)
def test_no_flat_polymorphic_paths(theme):
    hits = [f"{m.group(1)}.{m.group(2)}" for m in _FLAT_RE.finditer(theme.read_text())]
    assert not hits, f"{theme.name}: assign polymorphic fields whole-object, not flat-path: {hits}"
