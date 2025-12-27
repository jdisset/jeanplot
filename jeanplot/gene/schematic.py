from __future__ import annotations
from typing import Literal, TYPE_CHECKING
from pydantic import Field, PrivateAttr

from jeanplot.core.container import Container
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.models import LayoutConstraints, BoxStyle
from jeanplot.core.style import jstyle
from jeanplot.gene.data import CircuitData

if TYPE_CHECKING:
    from jeanplot.gene.elements import TranscriptionUnit


class TranscriptionUnitRow(Container):
    """Row of transcription units."""

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(direction="row", gap=20, align_items="center")
    )


class SourceAnnotation(Container):
    """Wrapper for TUs belonging to a source, with dashed border and marker tag."""

    source_id: str | None = None
    marker: str | None = None
    ratios: list[float] | None = None
    source_type: Literal["plasmid", "cotx", "mix", "linear"] | None = "cotx"
    style_class: list[str] = Field(default_factory=lambda: ["SourceAnnotation"])

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", gap=20, align_items="center", justify_content="start"
        )
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.marker or self.ratios:
            from jeanplot.gene.elements import Source

            source_proxy = Source(
                id=f"proxy_{self.id}" if self.id else None,
                source_type=self.source_type,
                marker=self.marker,
                ratios=self.ratios,
                parent=self,
            )
            jstyle.apply(source_proxy)
            tag_cont = getattr(source_proxy, "_tag_container", None)
            if tag_cont:
                if tag_cont in source_proxy.children:
                    source_proxy.children.remove(tag_cont)
                tag_cont.parent = self
                tag_cont.is_overlay = True
                self.add_child(tag_cont)


class GeneticSchematic(Container):
    """Grid-based genetic circuit schematic driven by CircuitData."""

    data: CircuitData
    grid_gap: tuple[float, float] = (40.0, 20.0)
    orientation: Literal["row", "column"] = "column"
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    show_sources: bool = True
    show_interactions: bool = True

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(direction="column", gap=20, align_items="start")
    )
    style: BoxStyle = Field(default_factory=lambda: BoxStyle(padding=(10, 10, 10, 10)))

    _tu_components: dict[str, TranscriptionUnit] = PrivateAttr(default_factory=dict)
    _connections: list[Connection] = PrivateAttr(default_factory=list)
    _grid_coords: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._build_layout()

    def _build_layout(self):
        self._assign_grid_positions()
        self._build_tus()
        if self.show_interactions:
            self._build_interactions()

    def _assign_grid_positions(self):
        """Assign grid row/col to each TU based on source grouping."""
        source_tu_map: dict[str, list[str]] = {}
        for source in self.data.sources:
            source_tu_map[source.id] = source.tu_ids

        orphan_tus = [
            tu.id
            for tu in self.data.transcription_units
            if not any(tu.id in ids for ids in source_tu_map.values())
        ]

        row = 0
        for source_id, tu_ids in source_tu_map.items():
            for col, tu_id in enumerate(tu_ids):
                self._grid_coords[tu_id] = (row, col)
            row += 1

        for col, tu_id in enumerate(orphan_tus):
            self._grid_coords[tu_id] = (row, col)

    def _build_tus(self):
        """Create TranscriptionUnit components from data."""
        from jeanplot.gene.elements import TranscriptionUnit, GeneticPart

        # Build source lookup: source_id -> (marker, type, ratios)
        source_by_tu: dict[str, str] = {}
        source_info: dict[str, tuple[str | None, str | None, list[float] | None]] = {}
        for source in self.data.sources:
            for tu_id in source.tu_ids:
                source_by_tu[tu_id] = source.id
            source_info[source.id] = (source.marker, source.source_type, source.ratios)

        rows_dict: dict[int, list[TranscriptionUnit]] = {}

        for tu_data in self.data.transcription_units:
            tu = TranscriptionUnit(
                id=tu_data.id,
                name=tu_data.name,
                ratio_percent=tu_data.ratio_percent,
                disabled=tu_data.disabled,
            )
            for part_data in tu_data.parts:
                part = GeneticPart.from_data(part_data)
                tu.add_child(part)
            self._tu_components[tu_data.id] = tu

            row, _ = self._grid_coords.get(tu_data.id, (0, 0))
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(tu)

        for row_idx in sorted(rows_dict.keys()):
            tus_in_row = rows_dict[row_idx]
            if not tus_in_row:
                continue

            # Check if this row's TUs belong to a source with marker/ratios
            first_tu_id = tus_in_row[0].id
            source_id = source_by_tu.get(first_tu_id)
            marker, source_type, ratios = source_info.get(source_id, (None, None, None)) if source_id else (None, None, None)

            if self.show_sources and (marker or ratios):
                wrapper = SourceAnnotation(
                    id=f"source_{source_id}",
                    source_id=source_id,
                    marker=marker,
                    ratios=ratios,
                    source_type=source_type or "cotx",
                    children=tus_in_row,
                )
                self.add_child(wrapper)
            else:
                row_container = TranscriptionUnitRow(
                    id=f"tu_row_{row_idx}",
                    children=tus_in_row,
                )
                self.add_child(row_container)

    def _build_interactions(self):
        from jeanplot.core.svg import LineEndFlat

        for interaction in self.data.interactions:
            source_tu = self._tu_components.get(interaction.source_tu)
            target_tu = self._tu_components.get(interaction.target_tu)
            if not source_tu or not target_tu:
                continue

            end_cap = LineEndFlat(stroke_width=1.5, length=8.0) if interaction.interaction_type in ("inhibition", "repression") else None
            curve = SimpleBezierCurve() if self.connection_style == "bezier" else OrthogonalCurve()

            conn = Connection(
                id=f"conn_{interaction.id}",
                start_component=f"//{interaction.source_tu}/{interaction.source_part}",
                end_component=f"//{interaction.target_tu}/{interaction.target_part}",
                curve_type=curve,
                end_cap=end_cap,
                is_overlay=True,
            )
            self._connections.append(conn)
            self.add_child(conn)

    @classmethod
    def from_circuit(cls, circuit: CircuitData, **kwargs) -> GeneticSchematic:
        """Factory method for creating schematic from circuit data."""
        return cls(data=circuit, **kwargs)
