"""Tests for genetic element components."""
import pytest
from jeanplot.gene.elements import (
    GeneticPart,
    Promoter,
    Terminator,
    ERN,
    FluoMarker,
    UorfGroup,
    ERN5pRecog,
    TranscriptionUnit,
    make_vertical_anchors,
)
from jeanplot.gene.data import PartData


class TestGeneticPartFromData:
    """GeneticPart.from_data() factory."""

    @pytest.mark.parametrize("role,expected_cls", [
        ("promoter", Promoter),
        ("terminator", Terminator),
        ("regulator", ERN),
        ("reporter", FluoMarker),
        ("uorf", UorfGroup),
        ("recognition_site", ERN5pRecog),
    ])
    def test_role_to_class_mapping(self, role, expected_cls):
        """Correct subclass instantiated for each role."""
        data = PartData(id="p1", name="Test", role=role)
        part = GeneticPart.from_data(data)
        assert isinstance(part, expected_cls)

    def test_unknown_role_uses_base(self):
        """Unknown role creates base GeneticPart."""
        data = PartData(id="p1", name="Test", role="cds")
        part = GeneticPart.from_data(data)
        assert type(part) is GeneticPart

    def test_id_preserved(self):
        """Part ID matches data ID."""
        data = PartData(id="my_promoter", name="pCMV", role="promoter")
        part = GeneticPart.from_data(data)
        assert part.id == "my_promoter"


class TestAutoLabel:
    """AutoLabelMixin behavior."""

    def test_ern_auto_label(self):
        """ERN creates label from part_name."""
        ern = ERN(id="ern1", part_name="CasE")
        assert ern.label is not None
        assert ern.label.text == "CasE"

    def test_fluo_auto_label(self):
        """FluoMarker creates label from part_name."""
        fluo = FluoMarker(id="f1", part_name="GFP")
        assert fluo.label is not None
        assert fluo.label.text == "GFP"

    def test_uorf_truncates_label(self):
        """UorfGroup truncates label at underscore."""
        uorf = UorfGroup(id="u1", part_name="2xuORF_v1")
        assert uorf.label is not None
        assert uorf.label.text == "2xuORF"

    def test_no_label_without_name(self):
        """No label created if part_name missing."""
        ern = ERN(id="ern1")
        assert ern.label is None

    def test_explicit_label_not_overwritten(self):
        """Explicit label preserved."""
        from jeanplot import Text
        custom = Text(id="custom", text="Custom Label")
        ern = ERN(id="ern1", part_name="CasE", label=custom)
        assert ern.label.text == "Custom Label"


class TestVerticalAnchors:
    """make_vertical_anchors helper."""

    def test_returns_two_anchors(self):
        """Returns top and bottom anchors."""
        anchors = make_vertical_anchors()
        assert len(anchors) == 2

    def test_anchor_directions(self):
        """Anchors point up and down."""
        anchors = make_vertical_anchors()
        directions = [a.direction for a in anchors]
        assert (0, 1) in directions  # top
        assert (0, -1) in directions  # bottom

    def test_prefix_applied(self):
        """Prefix prepended to anchor IDs."""
        anchors = make_vertical_anchors(prefix="my-")
        ids = [a.id for a in anchors]
        assert any("my-" in id for id in ids)


class TestTranscriptionUnit:
    """TranscriptionUnit component."""

    def test_tu_creates_line(self, mock_renderer):
        """TU creates internal line element."""
        tu = TranscriptionUnit(id="tu1", name="Test")
        assert tu._tu_line is not None

    def test_tu_name_creates_label(self):
        """TU name creates label component."""
        tu = TranscriptionUnit(id="tu1", name="Reporter")
        assert tu.label is not None
        assert tu.label.text == "Reporter"

    def test_tu_row_layout(self):
        """TU uses row layout for parts."""
        tu = TranscriptionUnit(id="tu1")
        assert tu.layout.direction == "row"

    def test_tu_children_are_parts(self, mock_renderer):
        """Parts added as children."""
        tu = TranscriptionUnit(id="tu1")
        prom = Promoter(id="p1", part_name="pCMV")
        tu.add_child(prom)
        tu.measure_and_layout(mock_renderer)

        # Promoter should be among layout children
        layout_children = tu._layout_children_cache
        assert prom in layout_children


class TestPartTypes:
    """Part type specific behavior."""

    def test_promoter_part_type(self):
        """Promoter has correct part_type."""
        p = Promoter(id="p1")
        assert p.part_type == "promoter"

    def test_terminator_part_type(self):
        """Terminator has correct part_type."""
        t = Terminator(id="t1")
        assert t.part_type == "terminator"

    def test_ern_has_anchors(self):
        """ERN has vertical anchors."""
        ern = ERN(id="e1")
        assert len(ern.anchor_points) == 2

    def test_ern5precog_has_anchors(self):
        """ERN5pRecog has vertical anchors."""
        recog = ERN5pRecog(id="r1")
        assert len(recog.anchor_points) == 2
