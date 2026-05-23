"""TUI surface tests: receipt, tree, spinner, inline preview."""

import io
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
    use_tui,
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
    with use_tui(tui):
        fig.render()
    assert sink.getvalue() == ""


def test_default_receipt(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(preview="off", console=console)
    with use_tui(tui):
        fig.render()
    out = sink.getvalue()
    assert "rendered" in out
    assert "Figure" in out
    assert "2 panels" in out
    assert "t.png" in out


def test_long_path_emits_clickable_link_without_breaking(tmp_path: Path):
    load_plot_theme()
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(50, 2)).astype(np.float32)
    y = rng.uniform(0, 1, size=(50, 1)).astype(np.float32)
    data = PlotData(xval=x, yval=y, input_names=["a", "b"], output_name="o")
    deep = tmp_path / "a_very_long_intermediate_subdirectory" / "another_one"
    deep.mkdir(parents=True)
    fig = Figure(
        SmoothPanel2D(plot_data=data),
        output_dir=str(deep),
        output_file="a_very_long_file_name_that_would_definitely_wrap.png",
    )
    sink = io.StringIO()
    narrow = Console(file=sink, force_terminal=True, width=60, highlight=False)
    with use_tui(RenderTUI(preview="off", console=narrow)):
        fig.render()
    out = sink.getvalue()
    assert out.count("\033]8;id=") == 1
    assert out.count("\033]8;;\033\\") == 1
    link_open = out.index("\033]8;id=")
    link_close = out.index("\033]8;;\033\\")
    assert "\n" not in out[link_open:link_close]


def test_verbose_emits_component_tree_and_span_tree(fig: Figure):
    console, sink = _string_console()
    tui = RenderTUI(verbose=True, preview="off", console=console)
    with use_tui(tui):
        fig.render()
    out = sink.getvalue()
    assert "Figure" in out
    assert "SmoothPanel2D" in out
    assert "render" in out or "more" in out


def test_numbered_span_renders_progress_bar():
    from dracon.progress import each, use_subscriber

    from jeanplot._tui import _bar, _parse_numbered

    console, _ = _string_console()
    tui = RenderTUI(quiet=False, preview="off", console=console)
    with use_subscriber(tui):
        for _ in each("predicting", list(range(4))):
            pass
    tui._stop_spinner()
    assert "predicting" in tui._last_label
    assert "█" in tui._last_label
    assert _parse_numbered("predicting 3/12") == ("predicting", 3, 12)
    assert _parse_numbered("plain name") is None
    assert "█" in _bar(0.5) and "░" in _bar(0.5)


def test_tree_aggregates_fast_siblings_under_slow_parent():
    import time as _time

    from dracon.progress import step, use_subscriber

    console, sink = _string_console()
    tui = RenderTUI(verbose=True, preview="off", console=console)
    with use_subscriber(tui):
        with step("slow parent"):
            _time.sleep(0.12)
            for i in range(5):
                with step(f"trivial {i}"):
                    pass
    tui._print_span_tree()
    out = sink.getvalue()
    assert "slow parent" in out
    assert "more" in out
    assert "trivial 0" not in out


def test_spans_captured(fig: Figure):
    tui = RenderTUI(quiet=True, preview="off")
    with use_tui(tui):
        fig.render()
    names = [s.name for s in tui._spans.values()]
    assert "render" in names
    assert "layout" in names
    assert "draw" in names
    assert "save" in names
    assert sum(1 for n in names if n.startswith("panel ")) == 2


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


def test_emit_kitty_uses_unicode_placeholders(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_kitty(sink, b"x" * 10, cols=4, rows=3)
    out = sink.getvalue()
    assert "\033_Ga=T,U=1,i=" in out
    assert "f=100,c=4,r=3,q=2" in out
    assert "\U0010eeee" in out
    assert "\033[38;2;" in out and "\033[39m" in out
    assert out.count("\U0010eeee") == 12


def test_emit_kitty_chunks_long_payload(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_kitty(sink, b"x" * 8000, cols=4, rows=3)
    out = sink.getvalue()
    transmission = out.split("\033[38;2;")[0]
    frames = transmission.split("\033_G")[1:]
    assert len(frames) >= 2
    assert frames[0].startswith("a=T,U=1,i=")
    assert ",m=1" in frames[0]
    for f in frames[1:-1]:
        assert f.startswith("m=1;")
    assert frames[-1].startswith("m=0;") or frames[-1].startswith("m=1;")


def test_emit_kitty_image_id_round_trips_through_fg_color(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    sink = io.StringIO()
    _emit_kitty(sink, b"hi", cols=2, rows=1)
    out = sink.getvalue()
    import re

    m_ctrl = re.search(r"i=(\d+)", out)
    m_fg = re.search(r"\033\[38;2;(\d+);(\d+);(\d+)m", out)
    assert m_ctrl and m_fg
    image_id = int(m_ctrl.group(1))
    r, g, b = (int(m_fg.group(i)) for i in (1, 2, 3))
    assert (r << 16) | (g << 8) | b == image_id


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
    _emit_kitty(sink, b"x" * 8000, cols=4, rows=3)
    out = sink.getvalue()
    assert out.count("\033Ptmux;") >= 2
    assert "\033\033_G" in out
    assert "\U0010eeee" in out


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
    with use_tui(tui):
        fig.render()
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
    with use_tui(RenderTUI(preview="off", console=console)):
        fig.render()
    sink.truncate(0)
    sink.seek(0)
    tui2 = RenderTUI(preview="off", console=Console(file=sink, force_terminal=True))
    with use_tui(tui2):
        fig.render(overwrite=False)
    assert sink.getvalue() == ""
