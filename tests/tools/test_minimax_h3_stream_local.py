from __future__ import annotations

import os
import stat
from pathlib import Path

from tools.base_tool import ToolStatus
from tools.video.minimax_h3_stream_local import MiniMaxH3StreamLocal


def _fake_h3stream(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_bytes(b'fake mp4')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_is_available_when_runner_and_model_dir_are_configured(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "h3stream"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _fake_h3stream(runner)
    monkeypatch.setenv("MINIMAX_H3_STREAM_PATH", str(runner))
    monkeypatch.setenv("MINIMAX_H3_MODEL_DIR", str(model_dir))

    assert MiniMaxH3StreamLocal().get_status() is ToolStatus.AVAILABLE


def test_generates_an_mp4_with_the_local_h3stream_runner(tmp_path: Path, monkeypatch) -> None:
    runner = tmp_path / "h3stream"
    model_dir = tmp_path / "models"
    output_path = tmp_path / "project" / "assets" / "video" / "robot.mp4"
    model_dir.mkdir()
    _fake_h3stream(runner)
    monkeypatch.setenv("MINIMAX_H3_STREAM_PATH", str(runner))
    monkeypatch.setenv("MINIMAX_H3_MODEL_DIR", str(model_dir))

    result = MiniMaxH3StreamLocal().execute(
        {
            "prompt": "A friendly robot walking through a sunny garden.",
            "output_path": str(output_path),
            "width": 512,
            "height": 288,
            "seconds": 5,
            "steps": 4,
            "seed": 42,
        }
    )

    assert result.success is True
    assert result.data["output_path"] == str(output_path)
    assert result.artifacts == [str(output_path)]
    assert result.cost_usd == 0.0
    assert output_path.read_bytes() == b"fake mp4"


def test_reports_unavailable_when_the_runner_is_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("MINIMAX_H3_STREAM_PATH", raising=False)
    monkeypatch.delenv("MINIMAX_H3_MODEL_DIR", raising=False)
    monkeypatch.delenv("H3STREAM_PATH", raising=False)

    assert MiniMaxH3StreamLocal().get_status() is ToolStatus.UNAVAILABLE
