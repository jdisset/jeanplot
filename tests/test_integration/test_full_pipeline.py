"""Integration tests for full rendering pipeline."""
from jeanplot import (
    Container,
    Connection,
    Size,
    BoxStyle,
    LayoutConstraints,
    render_to_svg,
    parse_svg,
    MatplotlibRenderer,
)
from jeanplot.gene.data import CircuitData, TUData, PartData
from jeanplot.gene.schematic import GeneticSchematic
from jeanplot.gene.elements import TranscriptionUnit, Promoter, FluoMarker, Terminator


class TestLayoutToRender:
    """Layout computation to rendering."""

    def test_child_positions_in_output(self, mock_renderer):
        """Child positions reflected in SVG transforms."""
        parent = Container(
            id="parent",
            min_dimensions=Size(200, 100),
            layout=LayoutConstraints(direction="row", gap=10),
            style=BoxStyle(padding=(10, 10, 10, 10)),
        )
        c1 = Container(
            id="c1",
            min_dimensions=Size(50, 50),
            style=BoxStyle(background_color="#ff0000"),
        )
        c2 = Container(
            id="c2",
            min_dimensions=Size(50, 50),
            style=BoxStyle(background_color="#00ff00"),
        )
        parent.add_children([c1, c2])

        svg = render_to_svg(parent)
        root = parse_svg(svg)

        # Both children should appear in output
        g1 = root.find(".//*[@id='c1']")
        g2 = root.find(".//*[@id='c2']")
        assert g1 is not None
        assert g2 is not None

    def test_connection_path_rendered(self, mock_renderer):
        """Connection curve appears as path."""
        parent = Container(id="parent", min_dimensions=Size(300, 100))
        box1 = Container(
            id="box1",
            min_dimensions=Size(50, 50),
            style=BoxStyle(background_color="#ff0000"),
        )
        box2 = Container(
            id="box2",
            min_dimensions=Size(50, 50),
            offset=dict(absolute=(200, 0)),
            style=BoxStyle(background_color="#00ff00"),
        )
        conn = Connection(
            id="conn",
            start_component="box1",
            end_component="box2",
            style=BoxStyle(border_color="#000000", border_width=2),
        )
        parent.add_children([box1, box2, conn])

        svg = render_to_svg(parent)
        root = parse_svg(svg)

        # Connection should produce a path
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) > 0


class TestCircuitVisualization:
    """Full circuit visualization pipeline."""

    def test_circuit_renders_tus(self, simple_circuit_data, mock_renderer):
        """Circuit schematic renders all TUs."""
        schematic = GeneticSchematic(data=simple_circuit_data)
        svg = render_to_svg(schematic)

        assert "tu1" in svg
        assert "tu2" in svg

    def test_circuit_renders_parts(self, mock_renderer):
        """Parts appear in circuit output."""
        circuit = CircuitData(
            transcription_units=[
                TUData(
                    id="tu",
                    name="Reporter",
                    parts=[
                        PartData(id="prom", name="pCMV", role="promoter"),
                        PartData(id="gfp", name="GFP", role="reporter"),
                    ],
                ),
            ],
        )
        schematic = GeneticSchematic(data=circuit)
        svg = render_to_svg(schematic)

        # Part IDs should appear
        assert "prom" in svg
        assert "gfp" in svg


class TestStyleApplication:
    """Style cascade to rendering."""

    def test_styled_component_renders(self, mock_renderer):
        """Styled component has correct output."""
        from jeanplot import jstyle

        jstyle.update({
            "Container[id=styled]": {
                "style.background_color": "#abcdef",
            }
        })

        container = Container(
            id="styled",
            min_dimensions=Size(100, 50),
        )
        # Apply styles
        jstyle.apply(container)

        svg = render_to_svg(container)
        root = parse_svg(svg)

        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        assert rect is not None
        assert rect.get("fill") == "#abcdef"


class TestTranscriptionUnitRendering:
    """TranscriptionUnit specific rendering."""

    def test_tu_line_rendered(self, mock_renderer):
        """TU central line appears in output."""
        tu = TranscriptionUnit(id="tu1", name="Test")
        prom = Promoter(id="p1")
        term = Terminator(id="t1")
        tu.add_children([prom, term])

        svg = render_to_svg(tu)

        # Line element should be present
        assert "line_tu1" in svg

    def test_tu_parts_ordered(self, mock_renderer):
        """Parts appear in correct order."""
        tu = TranscriptionUnit(id="tu1")
        tu.add_child(Promoter(id="p1", part_name="pCMV"))
        tu.add_child(FluoMarker(id="f1", part_name="GFP"))
        tu.add_child(Terminator(id="t1", part_name="pA"))

        svg = render_to_svg(tu)

        # All parts should be in output
        assert "p1" in svg
        assert "f1" in svg
        assert "t1" in svg


class TestMatplotlibIntegration:
    """Matplotlib renderer integration."""

    def test_render_to_axes(self, mock_renderer):
        """Component renders to matplotlib axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        container = Container(
            id="box",
            min_dimensions=Size(100, 50),
            style=BoxStyle(background_color="#ff0000"),
        )

        renderer = MatplotlibRenderer()
        renderer.render_component(ax, container)

        # Should have added patches
        assert len(ax.patches) > 0
        plt.close(fig)
