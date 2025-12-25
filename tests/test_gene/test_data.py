"""Tests for gene data models."""
import pytest
from pydantic import ValidationError
from jeanplot.gene.data import PartData, TUData, SourceData, InteractionData, CircuitData


class TestPartData:
    """PartData validation and construction."""

    def test_valid_roles(self):
        """All defined roles accepted."""
        valid_roles = [
            "promoter", "rbs", "cds", "terminator", "operator",
            "insulator", "origin", "regulator", "reporter", "uorf",
            "recognition_site",
        ]
        for role in valid_roles:
            part = PartData(id="p", name="part", role=role)
            assert part.role == role

    def test_invalid_role_rejected(self):
        """Unknown role raises validation error."""
        with pytest.raises(ValidationError):
            PartData(id="p", name="part", role="unknown_role")

    def test_orientation_default(self):
        """Default orientation is forward."""
        part = PartData(id="p", name="part", role="cds")
        assert part.orientation == "forward"

    def test_reverse_orientation(self):
        """Reverse orientation accepted."""
        part = PartData(id="p", name="part", role="cds", orientation="reverse")
        assert part.orientation == "reverse"


class TestTUData:
    """TUData construction."""

    def test_empty_parts_list(self):
        """TU can have empty parts list."""
        tu = TUData(id="tu1", name="TU1")
        assert tu.parts == []

    def test_parts_preserved(self, simple_tu_data):
        """Parts list preserved in order."""
        assert len(simple_tu_data.parts) == 3
        assert simple_tu_data.parts[0].role == "promoter"
        assert simple_tu_data.parts[1].role == "reporter"
        assert simple_tu_data.parts[2].role == "terminator"


class TestSourceData:
    """SourceData construction."""

    def test_source_types(self):
        """Valid source types accepted."""
        for st in ["plasmid", "linear", "mix"]:
            src = SourceData(id="s", source_type=st)
            assert src.source_type == st

    def test_default_plasmid(self):
        """Default source type is plasmid."""
        src = SourceData(id="s")
        assert src.source_type == "plasmid"


class TestInteractionData:
    """InteractionData construction."""

    def test_interaction_types(self):
        """Valid interaction types accepted."""
        for itype in ["inhibition", "activation", "cleavage", "sequestration"]:
            inter = InteractionData(
                id="i",
                source_tu="tu1",
                source_part="p1",
                target_tu="tu2",
                target_part="p2",
                interaction_type=itype,
            )
            assert inter.interaction_type == itype

    def test_default_inhibition(self):
        """Default interaction type is inhibition."""
        inter = InteractionData(
            id="i",
            source_tu="tu1",
            source_part="p1",
            target_tu="tu2",
            target_part="p2",
        )
        assert inter.interaction_type == "inhibition"


class TestCircuitData:
    """CircuitData construction."""

    def test_empty_circuit(self):
        """Empty circuit has empty lists."""
        circuit = CircuitData()
        assert circuit.transcription_units == []
        assert circuit.sources == []
        assert circuit.interactions == []

    def test_full_circuit(self, simple_circuit_data):
        """Circuit preserves all components."""
        assert len(simple_circuit_data.transcription_units) == 2
        assert len(simple_circuit_data.sources) == 2
        assert len(simple_circuit_data.interactions) == 1

    def test_metadata_dict(self):
        """Circuit accepts metadata dict."""
        circuit = CircuitData(metadata={"author": "test", "version": 1})
        assert circuit.metadata["author"] == "test"
