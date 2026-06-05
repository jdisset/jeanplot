from __future__ import annotations
from typing import Literal, NamedTuple, TYPE_CHECKING
from pydantic import Field, PrivateAttr

from jeanplot.core.container import Container
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.models import LayoutConstraints, BoxStyle
from jeanplot.core.style import jstyle
from jeanplot.gene.data import CircuitData

if TYPE_CHECKING:
    from jeanplot.gene.elements import TranscriptionUnit


class _SourceSummary(NamedTuple):
    marker: str | None
    source_type: str | None
    ratios: list[float] | None
    marker_ratio: float | None
    tag_label: str | None
    axis_tag: str | None


_EMPTY_SOURCE = _SourceSummary(None, None, None, None, None, None)


class TranscriptionUnitRow(Container):
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(direction="row", gap=20, align_items="center")
    )


class SourceAnnotation(Container):
    """Wrapper for TUs belonging to a source, with dashed border and marker tag."""

    source_id: str | None = None
    marker: str | None = None
    marker_ratio: float | None = None
    ratios: list[float] | None = None
    source_type: Literal["plasmid", "cotx", "mix", "linear"] | None = "cotx"
    tag_label: str | None = None
    axis_tag: str | None = None
    style_class: list[str] = Field(default_factory=lambda: ["SourceAnnotation"])

    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", gap=20, align_items="center", justify_content="start"
        )
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.axis_tag:
            from jeanplot.core.text import Text

            axis = Text(
                id=f"{self.id}_axis_tag" if self.id else None,
                text=self.axis_tag,
                style_class=["axis_tag"],
                is_overlay=True,
                parent=self,
            )
            self.add_child(axis)
        if self.marker or self.ratios or self.tag_label:
            from jeanplot.gene.elements import Source

            source_proxy = Source(
                id=f"proxy_{self.id}" if self.id else None,
                source_type=self.source_type,
                marker=self.marker,
                marker_ratio=self.marker_ratio,
                ratios=self.ratios,
                tag_label=self.tag_label,
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
        if self.orientation == "row" and "layout" not in self.model_fields_set:
            self.layout = LayoutConstraints(direction="row", gap=20, align_items="start")
        self._build_layout()

    def _build_layout(self):
        self._assign_grid_positions()
        self._build_tus()
        if self.show_interactions:
            self._build_interactions()

    def _assign_grid_positions(self):
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
        from jeanplot.gene.elements import TranscriptionUnit, GeneticPart

        source_by_tu: dict[str, str] = {}
        source_info: dict[str, _SourceSummary] = {}
        for source in self.data.sources:
            for tu_id in source.tu_ids:
                source_by_tu[tu_id] = source.id
            source_info[source.id] = _SourceSummary(
                source.marker,
                source.source_type,
                source.ratios,
                source.marker_ratio,
                source.tag_label,
                source.axis_tag,
            )

        rows_dict: dict[int, list[TranscriptionUnit]] = {}

        for tu_data in self.data.transcription_units:
            tu = TranscriptionUnit(
                id=tu_data.id,
                name=tu_data.name,
                ratio_normalized=tu_data.ratio_normalized,
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

            first_tu_id = tus_in_row[0].id
            source_id = source_by_tu.get(first_tu_id)
            info = source_info.get(source_id, _EMPTY_SOURCE) if source_id else _EMPTY_SOURCE

            if self.show_sources and (
                info.marker or info.ratios or info.tag_label or info.axis_tag
            ):
                wrapper = SourceAnnotation(
                    id=f"source_{source_id}",
                    source_id=source_id,
                    marker=info.marker,
                    marker_ratio=info.marker_ratio,
                    ratios=info.ratios,
                    source_type=info.source_type or "cotx",
                    tag_label=info.tag_label,
                    axis_tag=info.axis_tag,
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
            if not (
                self._tu_components.get(interaction.source_tu)
                and self._tu_components.get(interaction.target_tu)
            ):
                continue

            curve = SimpleBezierCurve() if self.connection_style == "bezier" else OrthogonalCurve()
            conn = Connection(
                id=f"conn_{interaction.id}",
                start_component=f"/**[id={interaction.source_tu}] > [id={interaction.source_part}]",
                end_component=f"/**[id={interaction.target_tu}] > [id={interaction.target_part}]",
                style_class=[f"interaction-{interaction.interaction_type}"],
                is_overlay=True,
            ).with_defaults(
                curve_type=curve,
                end_cap=LineEndFlat(stroke_width=1.5, length=8.0)
                if interaction.interaction_type in ("inhibition", "repression")
                else None,
            )
            self._connections.append(conn)
            self.add_child(conn)

    @classmethod
    def from_circuit(cls, circuit: CircuitData, **kwargs) -> GeneticSchematic:
        return cls(data=circuit, **kwargs)
