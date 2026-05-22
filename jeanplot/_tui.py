"""Terminal UI surface for jeanplot renders.

SSOT for the receipt block, the per-panel spinner, the component tree dump,
and inline image preview. Off unless the caller (typically the CLI) constructs
a `RenderTUI` and threads it through `render(..., tui=...)`.
"""

import base64
import io
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree

_KITTY_DIACRITICS = (
    0x0305,
    0x030D,
    0x030E,
    0x0310,
    0x0312,
    0x033D,
    0x033E,
    0x033F,
    0x0346,
    0x034A,
    0x034B,
    0x034C,
    0x0350,
    0x0351,
    0x0352,
    0x0357,
    0x035B,
    0x0363,
    0x0364,
    0x0365,
    0x0366,
    0x0367,
    0x0368,
    0x0369,
    0x036A,
    0x036B,
    0x036C,
    0x036D,
    0x036E,
    0x036F,
    0x0483,
    0x0484,
    0x0485,
    0x0486,
    0x0487,
    0x0592,
    0x0593,
    0x0594,
    0x0595,
    0x0597,
    0x0598,
    0x0599,
    0x059C,
    0x059D,
    0x059E,
    0x059F,
    0x05A0,
    0x05A1,
    0x05A8,
    0x05A9,
    0x05AB,
    0x05AC,
    0x05AF,
    0x05C4,
    0x0610,
    0x0611,
    0x0612,
    0x0613,
    0x0614,
    0x0615,
    0x0616,
    0x0617,
    0x0657,
    0x0658,
    0x0659,
    0x065A,
    0x065B,
    0x065D,
    0x065E,
    0x06D6,
    0x06D7,
    0x06D8,
    0x06D9,
    0x06DA,
    0x06DB,
    0x06DC,
    0x06DF,
    0x06E0,
    0x06E1,
    0x06E2,
    0x06E4,
    0x06E7,
    0x06E8,
    0x06EB,
    0x06EC,
    0x0730,
    0x0732,
    0x0733,
    0x0735,
    0x0736,
    0x073A,
    0x073D,
    0x073F,
    0x0740,
    0x0741,
    0x0743,
    0x0745,
    0x0747,
    0x0749,
    0x074A,
    0x07EB,
    0x07EC,
    0x07ED,
    0x07EE,
    0x07EF,
    0x07F0,
    0x07F1,
    0x07F3,
    0x0816,
    0x0817,
    0x0818,
    0x0819,
    0x081B,
    0x081C,
    0x081D,
    0x081E,
    0x081F,
    0x0820,
    0x0821,
    0x0822,
    0x0823,
    0x0825,
    0x0826,
    0x0827,
    0x0829,
    0x082A,
    0x082B,
    0x082C,
    0x082D,
    0x0951,
    0x0953,
    0x0954,
    0x0F82,
    0x0F83,
    0x0F86,
    0x0F87,
    0x135D,
    0x135E,
    0x135F,
    0x17DD,
    0x193A,
    0x1A17,
    0x1A75,
    0x1A76,
    0x1A77,
    0x1A78,
    0x1A79,
    0x1A7A,
    0x1A7B,
    0x1A7C,
    0x1B6B,
    0x1B6D,
    0x1B6E,
    0x1B6F,
    0x1B70,
    0x1B71,
    0x1B72,
    0x1B73,
    0x1CD0,
    0x1CD1,
    0x1CD2,
    0x1CDA,
    0x1CDB,
    0x1CE0,
    0x1DC0,
    0x1DC1,
    0x1DC3,
    0x1DC4,
    0x1DC5,
    0x1DC6,
    0x1DC7,
    0x1DC8,
    0x1DC9,
    0x1DCB,
    0x1DCC,
    0x1DD1,
    0x1DD2,
    0x1DD3,
    0x1DD4,
    0x1DD5,
    0x1DD6,
    0x1DD7,
    0x1DD8,
    0x1DD9,
    0x1DDA,
    0x1DDB,
    0x1DDC,
    0x1DDD,
    0x1DDE,
    0x1DDF,
    0x1DE0,
    0x1DE1,
    0x1DE2,
    0x1DE3,
    0x1DE4,
    0x1DE5,
    0x1DE6,
    0x1DFE,
    0x20D0,
    0x20D1,
    0x20D4,
    0x20D5,
    0x20D6,
    0x20D7,
    0x20DB,
    0x20DC,
    0x20E1,
    0x20E7,
    0x20E9,
    0x20F0,
    0x2CEF,
    0x2CF0,
    0x2CF1,
    0x2DE0,
    0x2DE1,
    0x2DE2,
    0x2DE3,
    0x2DE4,
    0x2DE5,
    0x2DE6,
    0x2DE7,
    0x2DE8,
    0x2DE9,
    0x2DEA,
    0x2DEB,
    0x2DEC,
    0x2DED,
    0x2DEE,
    0x2DEF,
    0x2DF0,
    0x2DF1,
    0x2DF2,
    0x2DF3,
    0x2DF4,
    0x2DF5,
    0x2DF6,
    0x2DF7,
    0x2DF8,
    0x2DF9,
    0x2DFA,
    0x2DFB,
    0x2DFC,
    0x2DFD,
    0x2DFE,
    0x2DFF,
    0xA66F,
    0xA67C,
    0xA67D,
    0xA6F0,
    0xA6F1,
    0xA8E0,
    0xA8E1,
    0xA8E2,
    0xA8E3,
    0xA8E4,
    0xA8E5,
    0xA8E6,
    0xA8E7,
    0xA8E8,
    0xA8E9,
    0xA8EA,
    0xA8EB,
    0xA8EC,
    0xA8ED,
    0xA8EE,
    0xA8EF,
    0xA8F0,
    0xA8F1,
    0xAAB0,
    0xAAB2,
    0xAAB3,
    0xAAB7,
    0xAAB8,
    0xAABE,
    0xAABF,
    0xAAC1,
    0xFE20,
    0xFE21,
    0xFE22,
    0xFE23,
    0xFE24,
    0xFE25,
    0xFE26,
    0x10A0F,
    0x10A38,
    0x1D185,
    0x1D186,
    0x1D187,
    0x1D188,
    0x1D189,
    0x1D1AA,
    0x1D1AB,
    0x1D1AC,
    0x1D1AD,
    0x1D242,
    0x1D243,
    0x1D244,
)
_KITTY_PLACEHOLDER = "\U0010eeee"

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


def _kitty_preview_cells(mfig: Any) -> tuple[int, int]:
    term = shutil.get_terminal_size((80, 24))
    cols = min(len(_KITTY_DIACRITICS), max(20, term.columns - 4))
    try:
        w_in, h_in = mfig.get_size_inches()
        aspect = h_in / max(w_in, 1e-6)
    except Exception:
        aspect = 0.5
    rows = max(8, min(len(_KITTY_DIACRITICS), term.lines - 6, int(cols * aspect / 2)))
    return cols, rows


def _kitty_placeholder_block(image_id: int, cols: int, rows: int) -> str:
    r, g, b = (image_id >> 16) & 0xFF, (image_id >> 8) & 0xFF, image_id & 0xFF
    fg = f"\033[38;2;{r};{g};{b}m"
    reset = "\033[39m"
    lines = []
    for row in range(rows):
        rmark = chr(_KITTY_DIACRITICS[row])
        cells = "".join(
            _KITTY_PLACEHOLDER + rmark + chr(_KITTY_DIACRITICS[col]) for col in range(cols)
        )
        lines.append(fg + cells + reset)
    return "\n".join(lines) + "\n"


def _emit_kitty(stream: Any, data: bytes, cols: int = 40, rows: int = 20) -> None:
    payload = base64.b64encode(data).decode()
    image_id = random.randint(1, 0xFFFFFF)
    chunk = 4096
    parts = [payload[i : i + chunk] for i in range(0, len(payload), chunk)]
    for i, part in enumerate(parts):
        more = i < len(parts) - 1
        if i == 0:
            ctrl = f"a=T,U=1,i={image_id},f=100,c={cols},r={rows},q=2" + (",m=1" if more else "")
        else:
            ctrl = "m=1" if more else "m=0"
        stream.write(_tmux_wrap(f"\033_G{ctrl};{part}\033\\"))
    stream.write(_kitty_placeholder_block(image_id, cols, rows))
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
            cols, rows = _kitty_preview_cells(self._mpl_figure)
            _emit_kitty(stream, data, cols=cols, rows=rows)
