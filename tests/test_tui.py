"""TUI surface tests: receipt, tree, spinner, inline preview."""

import io
import time
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from jeanplot import Figure, PlotData, SmoothPanel2D, load_plot_theme
from jeanplot._tui import (
    RenderTUI,
    _emit_iterm,
    _emit_kitty,
    _fmt_bytes,
    _fmt_dt,
    _preview_protocol,
    _tmux_wrap,
    build_tree,
)


@pytest.fixture
def fig(tmp_path: Path) -> Figure:
    load_plot_theme()
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(80, 2)).astype(np.float32)
    y = rng.uniform(0, 1, size=(80, 1)).astype(np.float32)
    data = PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")
    return Figure(
        SmoothPanel2D(plot_data=data, title="left"),
        SmoothPanel2D(plot_data=data, title="right"),
        output_dir=str(tmp_path),
        output_file="t.png",
    )


def _string_console() -> tuple[Console, io.StringIO]:
    sink = io.StringIO()
    return Console(file=sink, force_terminal=True, width=120, highlight=False), sink


def test_quiet_is_silent(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(quiet=True, preview="off", console=console)
    fig.render(tui=tui)
    assert sink.getvalue() == ""


def test_default_receipt(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(preview="off", console=console)
    fig.render(tui=tui)
    out = sink.getvalue()
    assert "rendered" in out
    assert "Figure" in out
    assert "2 panels" in out
    assert "t.png" in out
    assert "layout" in out and "draw" in out and "save" in out


def test_verbose_emits_tree_and_panel_breakdown(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(verbose=True, preview="off", console=console)
    fig.render(tui=tui)
    out = sink.getvalue()
    assert "Figure" in out
    assert "SmoothPanel2D" in out
    assert out.count("SmoothPanel2D") >= 3


def test_timings_recorded(fig: Figure):
    tui = RenderTUI(quiet=True, preview="off")
    fig.render(tui=tui)
    assert tui.timings.draw > 0
    assert tui.timings.save > 0
    assert len(tui.timings.panels) == 2
    assert all(dt > 0 for _, dt in tui.timings.panels)


def test_build_tree_shape(fig: Figure):
    fig.measure_and_layout(None)
    tree = build_tree(fig)
    assert "Figure" in tree.label
    assert len(tree.children) == 2


def test_label_only_shows_user_set_fields():
    from jeanplot._tui import _label

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(40, 2)).astype(np.float32)
    y = rng.uniform(0, 1, size=(40, 1)).astype(np.float32)
    data = PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")
    untitled = _label(SmoothPanel2D(plot_data=data))
    titled = _label(SmoothPanel2D(plot_data=data, title="hello"))
    assert "title" not in untitled
    assert "title" in titled and "hello" in titled


def test_fmt_helpers():
    assert _fmt_dt(0.0) == "0ms"
    assert _fmt_dt(0.5) == "500ms"
    assert _fmt_dt(2.5) == "2.50s"
    assert _fmt_bytes(0) == "0 B"
    assert _fmt_bytes(2048).endswith("KB")
    assert _fmt_bytes(5 * 1024 * 1024).endswith("MB")


def test_phase_accumulator():
    tui = RenderTUI(quiet=True, preview="off")
    with tui.phase("draw"):
        time.sleep(0.01)
    with tui.phase("draw"):
        time.sleep(0.01)
    assert tui.timings.draw >= 0.02


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
        ({"TERM_PROGRAM": "WezTerm"}, "iterm"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"GHOSTTY_RESOURCES_DIR": "/x"}, "kitty"),
        ({}, None),
    ],
)
def test_preview_protocol_detection(env, expected, monkeypatch):
    for k in ("TERM_PROGRAM", "TERM", "KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "TMUX"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _preview_protocol() == expected


def test_emit_iterm_encodes_payload(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_iterm(sink, b"hello")
    out = sink.getvalue()
    assert out.startswith("\033]1337;File=inline=1;")
    assert "aGVsbG8=" in out
    assert out.endswith("\a\n")


def test_emit_kitty_encodes_payload(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_kitty(sink, b"x" * 10)
    out = sink.getvalue()
    assert out.startswith("\033_Ga=T,f=100")
    assert out.endswith("\n")


def test_emit_kitty_chunks_long_payload(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_kitty(sink, b"x" * 8000)
    out = sink.getvalue()
    frames = out.rstrip("\n").split("\033_G")[1:]
    assert len(frames) >= 2
    assert frames[0].startswith("a=T,f=100,m=1;")
    for f in frames[1:-1]:
        assert f.startswith("m=1;")
    assert frames[-1].startswith("m=0;") or frames[-1].startswith("m=1;")


def test_preview_protocol_passes_through_tmux(monkeypatch):
    for k in ("TERM_PROGRAM", "TERM", "KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1234,0")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert _preview_protocol() == "iterm"


def test_tmux_wrap_outside_tmux_is_identity(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    assert _tmux_wrap("\033]1337;hello\a") == "\033]1337;hello\a"


def test_tmux_wrap_doubles_escapes(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/x,1,0")
    out = _tmux_wrap("\033]1337;hi\a")
    assert out.startswith("\033Ptmux;")
    assert out.endswith("\033\\")
    assert "\033\033]1337;hi\a" in out


def test_emit_iterm_wraps_inside_tmux(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/x,1,0")
    sink = io.StringIO()
    _emit_iterm(sink, b"hello")
    out = sink.getvalue()
    assert out.startswith("\033Ptmux;")
    assert "aGVsbG8=" in out
    assert out.rstrip("\n").endswith("\033\\")


def test_emit_kitty_wraps_each_chunk_inside_tmux(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/x,1,0")
    sink = io.StringIO()
    _emit_kitty(sink, b"x" * 8000)
    out = sink.getvalue()
    assert out.count("\033Ptmux;") >= 2
    assert "\033\033_G" in out


def test_phase_accepts_unknown_name():
    tui = RenderTUI(quiet=True, preview="off")
    with tui.phase("totally_new_phase"):
        time.sleep(0.005)
    assert getattr(tui.timings, "totally_new_phase") >= 0.005


def test_phase_finally_preserves_exception():
    tui = RenderTUI(quiet=True, preview="off")
    with pytest.raises(ValueError, match="user error"):
        with tui.phase("draw"):
            raise ValueError("user error")
    assert tui.timings.draw > 0


def test_preview_on_bypasses_tty_check(monkeypatch):
    for k in ("TERM_PROGRAM", "TERM", "KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "TMUX"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    sink = io.StringIO()
    console = Console(file=sink)
    assert console.is_terminal is False
    tui = RenderTUI(preview="on", console=console)
    tui._mpl_figure = object()
    assert tui._should_preview() is True
    tui2 = RenderTUI(preview="auto", console=console)
    tui2._mpl_figure = object()
    assert tui2._should_preview() is False


def test_receipt_survives_missing_output_file(fig: Figure, tmp_path: Path):
    console, sink = _string_console()
    tui = RenderTUI(preview="off", console=console)
    fig.render(tui=tui)
    out_path = Path(fig.output_dir) / fig.output_file  # type: ignore[arg-type]
    out_path.unlink()
    sink.truncate(0)
    sink.seek(0)
    tui.receipt(fig, out_path)
    out = sink.getvalue()
    assert "rendered" in out
    assert "→" not in out


def test_render_does_nothing_without_tui(fig: Figure, capsys):
    fig.render()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_overwrite_false_skips_when_file_exists(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(preview="off", console=console)
    fig.render(tui=tui)
    sink.truncate(0)
    sink.seek(0)
    tui2 = RenderTUI(preview="off", console=Console(file=sink, force_terminal=True))
    fig.render(tui=tui2, overwrite=False)
    assert sink.getvalue() == ""
