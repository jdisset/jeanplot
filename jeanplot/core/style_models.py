from __future__ import annotations

from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from jeanplot.core.style_selector import Selector, Specificity


class StyleRule(BaseModel):
    selector: Selector
    properties: dict[str, Any] = Field(default_factory=dict)
    nested_rules: dict[str, Any] = Field(default_factory=dict)
    is_context_rule: bool = False
    source_index: int = 0
    match_level: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class PropertyApplication(NamedTuple):
    specificity: Specificity
    is_context: bool
    mro_level: int
    source_order: int
    value: Any
