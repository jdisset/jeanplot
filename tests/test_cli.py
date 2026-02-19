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
