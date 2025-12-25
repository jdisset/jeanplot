# Core
from jeanplot.core.component import Component, AnchorComponent, Overlay
from jeanplot.core.container import Container
from jeanplot.core.models import Size, BoxStyle, LayoutConstraints, Offset, Transform, Shadow
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve, StraightCurve
from jeanplot.core.svg import LineEndFlat, LineEndCircle, LineEndArrow, SVGElement
from jeanplot.core.curve import CurveDefinition
from jeanplot.core.style import jstyle
from jeanplot.core.debug import (
    debug_print as debug_print,
    get_logger as get_logger,
    set_debug as set_debug,
    DebugMixin as DebugMixin,
)
from jeanplot.core.renderer import BaseRenderer, MatplotlibRenderer, SVGRenderer
from jeanplot.core.text import Text
from jeanplot.core.table import Table, TableRow, TableCell

# Testing utilities
from jeanplot.testing import (
    MockRenderer as MockRenderer,
    render_to_svg as render_to_svg,
    parse_svg as parse_svg,
    get_element_bounds as get_element_bounds,
    assert_element_position as assert_element_position,
    assert_element_size as assert_element_size,
    svg_hash as svg_hash,
)

# Gene visualization
from jeanplot.gene.elements import (
    GeneticPart,
    Promoter,
    Terminator,
    ERN,
    ERN5pRecog,
    FluoMarker,
    UorfGroup,
    TranscriptionUnit,
    Source,
)

# Deprecated (backward compatibility) - these will move to biocomptools
from jeanplot._deprecated.network_utils import TUInfo, Interaction
from jeanplot._deprecated.network_schematic import NetworkGeneticSchematic
from jeanplot._deprecated.network_diagram import (
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
    Overlay,
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
    SVGRenderer,
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
    SVGElement,
    Table,
    TableRow,
    TableCell,
    GeneticPart,
    Promoter,
    Terminator,
    ERN,
    ERN5pRecog,
    FluoMarker,
    UorfGroup,
    TranscriptionUnit,
    Source,
]

# --- Explicit model rebuilds after all types are defined ---
Component.model_rebuild(force=True)
AnchorComponent.model_rebuild(force=True)
Container.model_rebuild(force=True)
SVGElement.model_rebuild(force=True)
Connection.model_rebuild(force=True)
Text.model_rebuild(force=True)


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
