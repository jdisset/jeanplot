from jeanplot import cli


def test_theme_check_calls_loader_and_prints_ok(monkeypatch, capsys):
    called = {"value": False}

    def _fake_load_default_theme():
        called["value"] = True

    monkeypatch.setattr(cli, "load_default_theme", _fake_load_default_theme)

    rc = cli.main(["theme-check"])
    out = capsys.readouterr().out

    assert rc == 0
    assert called["value"] is True
    assert "theme ok" in out


def test_cli_without_command_prints_help(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage:" in out


def test_plot_job_invokes_yaml(tmp_path):
    """PlotJob.from_config + run() loads a YAML and renders to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from jeanplot import PlotData, Size
    from jeanplot.cli import PlotJob

    yaml = tmp_path / "tiny.yaml"
    yaml.write_text(
        """
<<(<): !include pkg:jeanplot:resources/themes/plots

!require plot_data: "..."
!set_default output_dir: "./"

figure: !Figure
  theme: !include pkg:jeanplot:resources/themes/plots.yaml@rules
  output_dir: ${output_dir}
  output_file: null
  layout: !LayoutConstraints { direction: row, gap: 8 }
  children:
    - !SmoothPanel1D
      plot_data: ${plot_data}
"""
    )
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(60, 1)).astype(np.float32)
    y = rng.uniform(0, 1, size=(60, 1)).astype(np.float32)
    pd = PlotData(xval=x, yval=y, input_names=["x0"], output_name="y")

    job = PlotJob.from_config(str(yaml), plot_data=pd, output_dir=str(tmp_path))
    job.figure.min_dimensions = Size(width=3.0, height=3.0)
    job.figure.dpi = 50
    fig = job.run()
    assert fig is job.figure
