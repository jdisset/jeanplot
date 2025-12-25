from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class PartData(BaseModel):
    id: str
    name: str
    role: Literal[
        "promoter",
        "rbs",
        "cds",
        "terminator",
        "operator",
        "insulator",
        "origin",
        "regulator",
        "reporter",
        "uorf",
        "recognition_site",
    ]
    orientation: Literal["forward", "reverse"] = "forward"
    sequence: str | None = None


class TUData(BaseModel):
    id: str
    name: str
    parts: list[PartData] = Field(default_factory=list)
    source_id: str | None = None
    position: int = 0


class SourceData(BaseModel):
    id: str
    name: str | None = None
    source_type: Literal["plasmid", "linear", "mix"] = "plasmid"
    tu_ids: list[str] = Field(default_factory=list)
    ratio: float | None = None
    marker: str | None = None


class InteractionData(BaseModel):
    id: str
    source_tu: str
    source_part: str
    target_tu: str
    target_part: str
    interaction_type: Literal["inhibition", "activation", "cleavage", "sequestration"] = (
        "inhibition"
    )


class CircuitData(BaseModel):
    transcription_units: list[TUData] = Field(default_factory=list)
    sources: list[SourceData] = Field(default_factory=list)
    interactions: list[InteractionData] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
