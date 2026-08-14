from __future__ import annotations

import sys
import types

from backlot import __main__ as cli


def test_cmd_serve_binds_the_requested_host(monkeypatch) -> None:
    calls: dict[str, object] = {}
    fake_uvicorn = types.ModuleType("uvicorn")

    def run(app: str, **kwargs: object) -> None:
        calls["app"] = app
        calls.update(kwargs)

    fake_uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    assert cli.cmd_serve(4751, host="0.0.0.0") == 0
    assert calls == {
        "app": "backlot.server:app",
        "host": "0.0.0.0",
        "port": 4751,
        "log_level": "warning",
    }
