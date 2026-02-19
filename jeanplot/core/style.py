"""Compatibility façade for the style engine.

Public imports stay stable (`jstyle`, `JStyle`, `Selector`, etc.) while
implementation now lives in dedicated modules.
"""

from jeanplot.core.style_engine import JStyle, jstyle
from jeanplot.core.style_models import PropertyApplication, StyleRule
from jeanplot.core.style_selector import Selector, Specificity

__all__ = [
    "Specificity",
    "Selector",
    "StyleRule",
    "PropertyApplication",
    "JStyle",
    "jstyle",
]
