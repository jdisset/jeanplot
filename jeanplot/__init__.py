from .component import Component, AnchorComponent
from .container import Container
from .models import Size, BoxStyle, LayoutConstraints, Offset, Transform, Shadow
from .connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from .svg import LineEndFlat, LineEndCircle, LineEndArrow
from .curve import CurveDefinition
from .network_utils import TUInfo, Interaction
from .style import jstyle
from .debug import debug_print, get_logger, set_debug
from .renderer import BaseRenderer
from .network_schematic import NetworkGeneticSchematic
from .matplotlib_renderer import MatplotlibRenderer
from .text import Text
from .network_diagram import (
    ComputeNode,
    TranscriptionNode,
    TranslationNode,
    AggregationNode,
    ERNNode,
    InvNode,
    FluoNode,
    DeadEndNode,
    TUNode,
    NetworkDiagram,
)

DEFAULT_TYPES = [
    Component,
    AnchorComponent,
    Container,
    Size,
    BoxStyle,
    LayoutConstraints,
    Offset,
    Transform,
    Shadow,
    Connection,
    OrthogonalCurve,
    SimpleBezierCurve,
    StraightCurve,
    LineEndFlat,
    LineEndCircle,
    LineEndArrow,
    CurveDefinition,
    TUInfo,
    Interaction,
    BaseRenderer,
    MatplotlibRenderer,
    NetworkGeneticSchematic,
    ComputeNode,
    TranscriptionNode,
    TranslationNode,
    AggregationNode,
    ERNNode,
    InvNode,
    FluoNode,
    DeadEndNode,
    TUNode,
    NetworkDiagram,
    Text,
]


def make_context_from_types(types):
    return {t.__name__: t for t in types}


def load_default_theme(force=False):
    _DEFAULT_THEME_PATH = "pkg:jeanplot:resources/themes/default.yaml"

    import dracon as dr

    _theme_dict = dr.load(
        _DEFAULT_THEME_PATH,
        enable_interpolation=True,
        raw_dict=True,
        context=make_context_from_types(DEFAULT_TYPES),
    )
    dr.resolve_all_lazy(_theme_dict)
    jstyle.clear()
    jstyle.update(_theme_dict)


load_default_theme()
