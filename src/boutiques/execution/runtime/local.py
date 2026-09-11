"""Local runtime: run the resolved argv directly via :mod:`subprocess`."""

from __future__ import annotations

from pathlib import Path

from boutiques.execution.runtime._subprocess import run_subprocess
from boutiques.execution.runtime.base import RunResult


def run(
    argv: list[str],
    *,
    container_image: object | None = None,  # ignored
    env: dict[str, str],
    cwd: Path,
    mounts: list[Path] | None = None,  # ignored
    runtime_args: list[str] | None = None,  # ignored
    stream: bool = True,
    capture: bool = True,
    image_path: Path | None = None,  # ignored
    no_pull: bool = False,  # ignored
) -> RunResult:
    """Run ``argv`` locally; stream/capture per the flags."""
    return run_subprocess(argv, env=env, cwd=cwd, stream=stream, capture=capture)
