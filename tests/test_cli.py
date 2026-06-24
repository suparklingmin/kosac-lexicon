import io
import json

import pytest

from kosac.cli import main


def test_cli_features(capsys):
    assert main(["features"]) == 0
    out = capsys.readouterr().out.split()
    assert "polarity" in out and "subjectivity-type" in out


def test_cli_citation(capsys):
    assert main(["citation"]) == 0
    assert capsys.readouterr().out.startswith("@misc{")


def test_cli_analyze_emits_json(capsys):
    pytest.importorskip("kiwipiepy")
    assert main(["analyze", "이 영화 정말 좋다"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "이 영화 정말 좋다"
    assert "polarity" in payload["features"]


def test_cli_analyze_reads_stdin(capsys, monkeypatch):
    pytest.importorskip("kiwipiepy")
    monkeypatch.setattr("sys.stdin", io.StringIO("좋다\n나쁘다\n"))
    assert main(["analyze", "--compact"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and all(json.loads(line)["features"] for line in lines)
