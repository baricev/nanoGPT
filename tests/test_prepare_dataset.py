import shutil
import runpy
from pathlib import Path

import requests


class DummyResponse:
    def __init__(self, text):
        self.text = text


def test_prepare_no_download(tmp_path, monkeypatch):
    data_dir = tmp_path / "shakespeare_char"
    shutil.copytree("data/shakespeare_char", data_dir)
    script = data_dir / "prepare.py"
    monkeypatch.setattr(requests, "get", lambda url: DummyResponse("abc" * 100))
    runpy.run_path(str(script))
    assert (data_dir / "train.bin").stat().st_size > 0
    assert (data_dir / "val.bin").stat().st_size > 0
    assert (data_dir / "meta.pkl").stat().st_size > 0
