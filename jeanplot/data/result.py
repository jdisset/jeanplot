from typing import Any


class PlotFunctionResult:
    """Return type for plot functions that want to attach computed metadata.

    Tuple-unpackable for backward compat: `a, b = plot_func(...)` still works.
    """

    __slots__ = ("rendering", "metadata")

    def __init__(self, rendering: Any, metadata: dict[str, Any] | None = None):
        self.rendering = rendering
        self.metadata = metadata or {}

    def __iter__(self):
        return iter(self.rendering)

    def __len__(self):
        return len(self.rendering)

    def __getitem__(self, idx):
        return self.rendering[idx]
