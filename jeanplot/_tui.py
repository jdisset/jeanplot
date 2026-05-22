"""Terminal UI surface for jeanplot renders.

SSOT for the receipt block, the per-panel spinner, the component tree dump,
and inline image preview. Off unless the caller (typically the CLI) constructs
a `RenderTUI` and threads it through `render(..., tui=...)`.
"""

import base64
import io
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree

_INTERESTING_FIELDS = (
    "title",
    "vlims",
    "xlims",
    "ylims",
    "cmap",
    "palette",
    "layout",
    "axes_size",
    "min_dimensions",
    "output_file",
)


def _short(value: Any) -> str:
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        text = type(value).__name__
    elif isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict, BaseModel))
    ):
        try:
            text = "[" + ", ".join(repr(x) for x in value) + "]"
        except TypeError:
            text = repr(value)
    else:
        text = repr(value)
    return text if len(text) <= 36 else text[:33] + "..."


def _label(component: Any) -> str:
    name = type(component).__name__
    set_fields = getattr(component, "_user_set_fields", None) or set()
    bits = []
    for f in _INTERESTING_FIELDS:
        if f not in set_fields:
            continue
        v = getattr(component, f, None)
        if v is None or v == "":
            continue
        bits.append(f"{f}={_short(v)}")
    return f"[bold]{name}[/]  [dim]{' · '.join(bits)}[/]" if bits else f"[bold]{name}[/]"


def _add_children(node: Tree, component: Any) -> None:
    for child in getattr(component, "children", None) or []:
        _add_children(node.add(_label(child)), child)


def build_tree(figure: Any) -> Tree:
    root = Tree(_label(figure))
    _add_children(root, figure)
    return root


def _fmt_dt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _osc8(path: Path, label: str | None = None) -> str:
    url = path.resolve().as_uri()
    text = label or str(path)
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def _preview_protocol() -> str | None:
    env = os.environ
    if env.get("TERM_PROGRAM") == "iTerm.app":
        return "iterm"
    if env.get("TERM_PROGRAM") == "WezTerm":
        return "iterm"
    if env.get("TERM", "").startswith("xterm-kitty") or "KITTY_WINDOW_ID" in env:
        return "kitty"
    if "GHOSTTY_RESOURCES_DIR" in env:
        return "kitty"
    return None


def _tmux_wrap(seq: str) -> str:
    if "TMUX" not in os.environ:
        return seq
    return "\033Ptmux;" + seq.replace("\033", "\033\033") + "\033\\"


def _png_bytes(mfig: Any, dpi: int = 96) -> bytes:
    buf = io.BytesIO()
    mfig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _emit_iterm(stream: Any, data: bytes) -> None:
    payload = base64.b64encode(data).decode()
    seq = f"\033]1337;File=inline=1;preserveAspectRatio=1:{payload}\a"
    stream.write(_tmux_wrap(seq) + "\n")
    stream.flush()


def _emit_kitty(stream: Any, data: bytes) -> None:
    payload = base64.b64encode(data).decode()
    chunk = 4096
    parts = [payload[i : i + chunk] for i in range(0, len(payload), chunk)]
    for i, part in enumerate(parts):
        more = i < len(parts) - 1
        if i == 0:
            ctrl = "a=T,f=100" + (",m=1" if more else "")
        else:
            ctrl = "m=1" if more else "m=0"
        stream.write(_tmux_wrap(f"\033_G{ctrl};{part}\033\\"))
    stream.write("\n")
    stream.flush()


@dataclass
class PhaseTimings:
    layout: float = 0.0
    draw: float = 0.0
    save: float = 0.0
    panels: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.layout + self.draw + self.save


@dataclass
class RenderTUI:
    verbose: bool = False
    quiet: bool = False
    preview: str = "auto"
    console: Console | None = None
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    _mpl_figure: Any = None
    _progress: Progress | None = None
    _task_id: int | None = None
    _panel_started_at: float = 0.0

    def __post_init__(self):
        if self.console is None:
            self.console = Console(stderr=True, highlight=False)

    @property
    def silent(self) -> bool:
        return self.quiet

    def show_tree(self, figure: Any) -> None:
        if self.silent or not self.verbose:
            return
        assert self.console is not None
        self.console.print(build_tree(figure))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            prior = getattr(self.timings, name, 0.0)
            setattr(self.timings, name, prior + time.perf_counter() - t0)

    @contextmanager
    def panels(self, total: int) -> Iterator["RenderTUI"]:
        assert self.console is not None
        if self.silent or total == 0 or not self.console.is_terminal:
            yield self
            return
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            self._progress = progress
            self._task_id = progress.add_task("rendering", total=total)
            try:
                yield self
            finally:
                self._progress = None
                self._task_id = None

    def panel_start(self, name: str) -> None:
        self._panel_started_at = time.perf_counter()
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=f"rendering {name}")

    def panel_done(self, name: str) -> None:
        dt = time.perf_counter() - self._panel_started_at
        self.timings.panels.append((name, dt))
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)

    def attach_mpl_figure(self, mfig: Any) -> None:
        self._mpl_figure = mfig

    def receipt(self, figure: Any, output_path: Path | None) -> None:
        if self.silent:
            return
        assert self.console is not None
        n_panels = len(self.timings.panels)
        w, h = (figure._dimensions.width, figure._dimensions.height)
        head = f"rendered [bold]{type(figure).__name__}[/] · {n_panels} panels · {w:g}×{h:g} in"
        size = None
        if output_path is not None:
            try:
                size = output_path.stat().st_size
            except OSError:
                size = None
        if size is not None:
            head += f" → {_fmt_bytes(size)}"
            self.console.print(head)
            self.console.print(f"  {_osc8(output_path)}")
        else:
            self.console.print(head)
        t = self.timings
        phases = [
            f"layout {_fmt_dt(t.layout)}",
            f"draw {_fmt_dt(t.draw)}",
            f"save {_fmt_dt(t.save)}",
        ]
        self.console.print(f"  [dim]{' · '.join(phases)}[/]")
        if self.verbose and t.panels:
            for name, dt in t.panels:
                self.console.print(f"    [dim]{name}  {_fmt_dt(dt)}[/]")
        if self._should_preview():
            self._inline_preview()

    def _should_preview(self) -> bool:
        if self.preview == "off" or self._mpl_figure is None:
            return False
        if _preview_protocol() is None:
            return False
        if self.preview == "on":
            return True
        assert self.console is not None
        return self.console.is_terminal

    def _inline_preview(self) -> None:
        proto = _preview_protocol()
        if proto is None:
            return
        data = _png_bytes(self._mpl_figure)
        stream = sys.stderr
        if proto == "iterm":
            _emit_iterm(stream, data)
        elif proto == "kitty":
            _emit_kitty(stream, data)
