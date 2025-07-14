import sys


def test_configurator_overrides(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "conf.py"
    cfg.write_text("batch_size = 2\nlearning_rate = 0.1\n")

    globs = {"batch_size": 1, "learning_rate": 0.01}
    monkeypatch.setattr(sys, "argv", ["configurator.py", str(cfg), "--batch_size=3"])
    exec(open("configurator.py").read(), globs)
    out = capsys.readouterr().out
    assert "Overriding config with" in out
    assert "Overriding: batch_size = 3" in out
    assert globs["batch_size"] == 3
    assert globs["learning_rate"] == 0.1
