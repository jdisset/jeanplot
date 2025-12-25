"""Tests for GeneticSchematic component."""
from jeanplot.gene.schematic import GeneticSchematic
from jeanplot.gene.data import CircuitData, TUData, PartData, SourceData, InteractionData


class TestSchematicCreation:
    """GeneticSchematic construction."""

    def test_from_empty_circuit(self):
        """Schematic from empty circuit has no children."""
        circuit = CircuitData()
        schematic = GeneticSchematic(data=circuit)
        # Only internal setup children, no TU rows
        assert len(schematic.children) == 0

    def test_from_circuit_factory(self, simple_circuit_data):
        """from_circuit factory method works."""
        schematic = GeneticSchematic.from_circuit(simple_circuit_data)
        assert schematic.data is simple_circuit_data


class TestTULayout:
    """TU arrangement in schematic."""

    def test_tus_created(self, simple_circuit_data, mock_renderer):
        """TU components created from data."""
        schematic = GeneticSchematic(data=simple_circuit_data)
        schematic.measure_and_layout(mock_renderer)

        assert "tu1" in schematic._tu_components
        assert "tu2" in schematic._tu_components

    def test_tus_have_parts(self, simple_circuit_data, mock_renderer):
        """TU components contain parts."""
        schematic = GeneticSchematic(data=simple_circuit_data)
        schematic.measure_and_layout(mock_renderer)

        tu1 = schematic._tu_components["tu1"]
        # TU should have children (parts + label + line)
        assert len(tu1.children) > 0


class TestGridPositioning:
    """Grid-based TU positioning."""

    def test_source_grouping(self, mock_renderer):
        """TUs grouped by source in grid."""
        circuit = CircuitData(
            transcription_units=[
                TUData(id="tu1", name="TU1", parts=[]),
                TUData(id="tu2", name="TU2", parts=[]),
            ],
            sources=[
                SourceData(id="src1", tu_ids=["tu1", "tu2"]),
            ],
        )
        schematic = GeneticSchematic(data=circuit)

        # Both TUs in same row (from same source)
        assert schematic._grid_coords["tu1"][0] == schematic._grid_coords["tu2"][0]

    def test_orphan_tus_grouped(self, mock_renderer):
        """TUs without source grouped together."""
        circuit = CircuitData(
            transcription_units=[
                TUData(id="tu1", name="TU1", parts=[]),
                TUData(id="tu2", name="TU2", parts=[]),
            ],
            sources=[],
        )
        schematic = GeneticSchematic(data=circuit)

        # Both should be in same row
        assert schematic._grid_coords["tu1"][0] == schematic._grid_coords["tu2"][0]


class TestInteractions:
    """Interaction visualization."""

    def test_interactions_create_connections(self, simple_circuit_data, mock_renderer):
        """Interactions generate connection components."""
        schematic = GeneticSchematic(data=simple_circuit_data, show_interactions=True)
        schematic.measure_and_layout(mock_renderer)

        assert len(schematic._connections) == 1

    def test_no_connections_when_disabled(self, simple_circuit_data, mock_renderer):
        """No connections when show_interactions=False."""
        schematic = GeneticSchematic(data=simple_circuit_data, show_interactions=False)
        schematic.measure_and_layout(mock_renderer)

        assert len(schematic._connections) == 0

    def test_invalid_interaction_skipped(self, mock_renderer):
        """Interaction with missing TU skipped."""
        circuit = CircuitData(
            transcription_units=[
                TUData(id="tu1", name="TU1", parts=[PartData(id="p1", name="X", role="cds")]),
            ],
            interactions=[
                InteractionData(
                    id="bad",
                    source_tu="tu1",
                    source_part="p1",
                    target_tu="nonexistent",
                    target_part="p2",
                ),
            ],
        )
        schematic = GeneticSchematic(data=circuit, show_interactions=True)
        schematic.measure_and_layout(mock_renderer)

        assert len(schematic._connections) == 0


class TestSchematicOptions:
    """Schematic configuration options."""

    def test_grid_gap_applied(self):
        """Custom grid gap stored."""
        circuit = CircuitData()
        schematic = GeneticSchematic(data=circuit, grid_gap=(50.0, 30.0))
        assert schematic.grid_gap == (50.0, 30.0)

    def test_default_connection_style(self, simple_circuit_data):
        """Default connection style is orthogonal."""
        schematic = GeneticSchematic(data=simple_circuit_data)
        assert schematic.connection_style == "orthogonal"
