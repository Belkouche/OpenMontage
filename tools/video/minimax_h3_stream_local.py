"""Local MiniMax-H3 generation through the h3stream Apple-Silicon runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class MiniMaxH3StreamLocal(BaseTool):
    """Generate a local MiniMax-H3 clip with AIXF666/minimax-h3-stream-mac."""

    name = "minimax_h3_stream_local"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "minimax_h3_stream"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = (
        "Install AIXF666/minimax-h3-stream-mac, then configure:\n"
        "  MINIMAX_H3_STREAM_PATH=/path/to/.venv/bin/h3stream\n"
        "  MINIMAX_H3_MODEL_DIR=/path/to/minimax-h3-stream-mac/models\n"
        "For Apple Silicon, start with 512x288, 5 seconds, 4 steps, and 24 GiB."
    )
    agent_skills = ["minimax-h3", "ai-video-gen"]
    capabilities = ["text_to_video"]
    supports = {
        "text_to_video": True,
        "offline": True,
        "native_audio": True,
        "local_gpu": True,
        "apple_silicon": True,
    }
    best_for = [
        "offline MiniMax-H3 clips on Apple Silicon",
        "selected hero shots in otherwise local OpenMontage productions",
    ]
    not_good_for = ["rapid iteration", "high-volume clip generation", "image-conditioned generation"]
    fallback_tools = ["wan_video", "ltx_video_local", "cogvideo_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "output_path"],
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "output_path": {"type": "string"},
            "run_dir": {"type": "string"},
            "width": {"type": "integer", "minimum": 128, "default": 512},
            "height": {"type": "integer", "minimum": 128, "default": 288},
            "seconds": {"type": "integer", "minimum": 1, "default": 5},
            "steps": {"type": "integer", "minimum": 1, "default": 4},
            "seed": {"type": "integer"},
            "audio": {"type": "boolean", "default": False},
            "memory_limit_gib": {"type": "number", "minimum": 1, "default": 24},
            "timeout_seconds": {"type": "number", "minimum": 1, "default": 7200},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=24576, vram_mb=24576, disk_mb=8000, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["prompt", "width", "height", "seconds", "steps", "seed", "audio"]
    side_effects = ["writes video file to output_path", "writes H3 artifacts to run_dir"]
    user_visible_verification = ["Watch generated clip for motion coherence and prompt adherence"]

    @staticmethod
    def _runner_path() -> Path | None:
        configured = os.environ.get("MINIMAX_H3_STREAM_PATH") or os.environ.get("H3STREAM_PATH")
        if configured:
            return Path(configured).expanduser()
        discovered = shutil.which("h3stream")
        return Path(discovered) if discovered else None

    @staticmethod
    def _model_dir() -> Path | None:
        configured = os.environ.get("MINIMAX_H3_MODEL_DIR")
        return Path(configured).expanduser() if configured else None

    def get_status(self) -> ToolStatus:
        runner = self._runner_path()
        model_dir = self._model_dir()
        if runner and runner.is_file() and os.access(runner, os.X_OK) and model_dir and model_dir.is_dir():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        steps = int(inputs.get("steps", 4))
        seconds = int(inputs.get("seconds", 5))
        return float(150 * steps * max(seconds, 1) / 5)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() is not ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Local MiniMax-H3 is unavailable. " + self.install_instructions)

        runner = self._runner_path()
        model_dir = self._model_dir()
        assert runner is not None and model_dir is not None
        output_path = Path(str(inputs["output_path"])).expanduser()
        run_dir = Path(str(inputs.get("run_dir") or output_path.parent / f"{output_path.stem}-h3-run"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(runner),
            "generate",
            "--model-dir",
            str(model_dir),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
            "--prompt",
            str(inputs["prompt"]),
            "--width",
            str(inputs.get("width", 512)),
            "--height",
            str(inputs.get("height", 288)),
            "--seconds",
            str(inputs.get("seconds", 5)),
            "--steps",
            str(inputs.get("steps", 4)),
            "--memory-limit-gib",
            str(inputs.get("memory_limit_gib", 24)),
            "--i-understand-experimental",
        ]
        if inputs.get("seed") is not None:
            command.extend(["--seed", str(inputs["seed"])])
        if inputs.get("audio", False):
            command.append("--audio")

        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(inputs.get("timeout_seconds", 7200)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Local MiniMax-H3 timed out after {inputs.get('timeout_seconds', 7200)} seconds. Run directory: {run_dir}",
                duration_seconds=round(time.monotonic() - start, 2),
            )

        duration = round(time.monotonic() - start, 2)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            return ToolResult(
                success=False,
                error=f"Local MiniMax-H3 generation failed (exit {completed.returncode}): {detail}",
                duration_seconds=duration,
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            return ToolResult(
                success=False,
                error=f"Local MiniMax-H3 exited successfully but did not create {output_path}.",
                duration_seconds=duration,
            )

        return ToolResult(
            success=True,
            data={
                "output_path": str(output_path),
                "run_dir": str(run_dir),
                "runner": str(runner),
                "model_dir": str(model_dir),
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=duration,
            seed=inputs.get("seed"),
            model="MiniMax-H3 (local h3stream)",
        )
