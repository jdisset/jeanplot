from jeanplot.core.component import Component, AnchorComponent, Overlay
from jeanplot.core.container import Container
from jeanplot.core.models import (
    Size,
    BoxStyle,
    LayoutConstraints,
    Offset,
    Transform,
    Shadow,
    TextHalo,
)
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
from jeanplot.core.connection_label import ConnectionLabel
from jeanplot.core.table import Table, TableRow, TableCell
from jeanplot.render import (
    render as render,
    render_to_string as render_to_string,
)

from jeanplot.testing import (
    MockRenderer as MockRenderer,
    render_to_svg as render_to_svg,
    parse_svg as parse_svg,
    get_element_bounds as get_element_bounds,
    assert_element_position as assert_element_position,
    assert_element_size as assert_element_size,
    svg_hash as svg_hash,
)

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
from jeanplot.gene.schematic import GeneticSchematic, SourceAnnotation
from jeanplot.gene.data import CircuitData, TUData, PartData, SourceData, InteractionData

from jeanplot.data import (
    PlotData,
    LazyPlotData,
    DataDimensions,
    PlotFunctionResult as PlotFunctionResult,
    Rescaler as Rescaler,
    IdentityRescaler as IdentityRescaler,
    GridData as GridData,
    extract_grid_data as extract_grid_data,
    grid_data_to_b64 as grid_data_to_b64,
    grid_data_from_b64 as grid_data_from_b64,
)
from jeanplot.color import (
    load_palettes,
    register_palettes,
    closest_name as closest_name,
)
from jeanplot.panels import (
    PlotPanel,
    Colorbar,
    Figure,
    SmoothPanel1D,
    SmoothPanel2D,
    SmoothGradMagnitudePanel2D,
    GradientFieldPanel2D,
    SmoothPanel3D,
    CubeView,
    MVPPanel,
    DensityPanel1D,
    GridHistogramPanel,
    ScatterPanel3D,
    ViolinPanel,
    ParticlePanel,
    StackedPolyPanel,
    AsciiHeatmapPanel,
    IdentityLineOverlay,
    DiagonalPathOverlay,
    SliceOverlay,
    SliceChordOverlay,
    AdditionVsRemovalOverlay,
    DensityContourOverlay,
    auto_panel as auto_panel,
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
    TextHalo,
    Connection,
    OrthogonalCurve,
    SimpleBezierCurve,
    StraightCurve,
    LineEndFlat,
    LineEndCircle,
    LineEndArrow,
    CurveDefinition,
    BaseRenderer,
    MatplotlibRenderer,
    SVGRenderer,
    Text,
    ConnectionLabel,
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
    GeneticSchematic,
    SourceAnnotation,
    CircuitData,
    TUData,
    PartData,
    SourceData,
    InteractionData,
    PlotData,
    LazyPlotData,
    DataDimensions,
    IdentityRescaler,
    PlotPanel,
    Colorbar,
    Figure,
    SmoothPanel1D,
    SmoothPanel2D,
    SmoothGradMagnitudePanel2D,
    GradientFieldPanel2D,
    SmoothPanel3D,
    CubeView,
    MVPPanel,
    DensityPanel1D,
    GridHistogramPanel,
    ScatterPanel3D,
    ViolinPanel,
    ParticlePanel,
    StackedPolyPanel,
    AsciiHeatmapPanel,
    IdentityLineOverlay,
    DiagonalPathOverlay,
    SliceOverlay,
    SliceChordOverlay,
    AdditionVsRemovalOverlay,
    DensityContourOverlay,
]

register_palettes(load_palettes("pkg:jeanplot:resources/colors/bio_palettes.yaml"))

SVGElement.model_rebuild(force=True)
Connection.model_rebuild(force=True)
Text.model_rebuild(force=True)
ConnectionLabel.model_rebuild(force=True)


def make_context_from_types(types):
    return {t.__name__: t for t in types}


from jeanplot.compose import (  # noqa: E402  (depends on DEFAULT_TYPES being declared)
    COMPOSE_HELPERS,
    panel_row as panel_row,
    panel_grid as panel_grid,
    panels_from_datas as panels_from_datas,
    build_figure_metadata as build_figure_metadata,
    default_output_name as default_output_name,
)


def make_plot_context(extra_types: list | None = None, extra: dict | None = None) -> dict:
    """Standard context for loading jeanplot YAML: types + compose helpers."""
    ctx = make_context_from_types(DEFAULT_TYPES + (extra_types or []))
    ctx.update(COMPOSE_HELPERS)
    if extra:
        ctx.update(extra)
    return ctx


_DEFAULT_THEME_CACHE = None


def load_default_theme(force: bool = False):
    global _DEFAULT_THEME_CACHE
    if _DEFAULT_THEME_CACHE is None or force:
        import dracon as dr

        cfg = dr.load(
            "pkg:jeanplot:resources/themes/default.yaml",
            enable_interpolation=True,
            raw_dict=True,
            context=make_context_from_types(DEFAULT_TYPES),
        )
        dr.resolve_all_lazy(cfg, except_for={"component"})
        _DEFAULT_THEME_CACHE = cfg["rules"]
    jstyle.clear()
    jstyle.update(_DEFAULT_THEME_CACHE)


def load_plot_theme(*extras: str) -> None:
    """Notebook/test escape hatch: layer plot theme files into the ambient jstyle.

    Production code should set `Figure.theme` instead.
    """
    import dracon as dr

    layers = [
        "pkg:jeanplot:resources/themes/default.yaml",
        "pkg:jeanplot:resources/themes/plots.yaml",
        *extras,
    ]
    loader = dr.DraconLoader(
        enable_interpolation=True,
        context=make_plot_context(),
    )
    cfg = loader.stack(*layers).construct()
    dr.resolve_all_lazy(cfg, except_for={"component"})
    jstyle.clear()
    jstyle.update(cfg["rules"])
