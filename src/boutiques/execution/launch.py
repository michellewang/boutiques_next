"""High-level ``launch`` orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from boutiques.execution.mounts import collect_file_mounts
from boutiques.execution.outputs import (
    ResolvedOutput,
    resolve_output_paths,
    resolve_stdio_outputs,
)
from boutiques.execution.resolve import resolve
from boutiques.execution.runtime import docker as _docker
from boutiques.execution.runtime import local as _local
from boutiques.execution.runtime import singularity as _singularity
from boutiques.execution.runtime.base import RunResult, RuntimeError_
from boutiques.loader import AnyDescriptor
from boutiques.models.v05.containers import RootfsImage

_RUNTIMES = {
    "local": _local,
    "docker": _docker,
    "singularity": _singularity,
}


@dataclass
class LaunchResult:
    """Outcome of a launch: what was run, what came back, what was declared."""

    command: list[str]
    runtime: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    outputs: list[ResolvedOutput] = field(default_factory=list)


def launch(
    descriptor: AnyDescriptor,
    invocation: dict[str, Any],
    *,
    runtime: str = "local",
    cwd: Path | None = None,
    runtime_args: list[str] | None = None,
    stream: bool = True,
    capture: bool = True,
    image_path: Path | None = None,
    no_pull: bool = False,
) -> LaunchResult:
    """Resolve the invocation and run the tool under the chosen runtime."""
    if runtime not in _RUNTIMES:
        raise RuntimeError_(f"Unknown runtime {runtime!r}. Known: {', '.join(sorted(_RUNTIMES))}.")
    if image_path is not None:
        if runtime != "singularity":
            raise RuntimeError_(
                f"--imagepath only applies to the singularity runtime (runtime={runtime!r})."
            )
        if isinstance(descriptor.container_image, RootfsImage):
            raise RuntimeError_(
                "--imagepath cannot override a rootfs container-image, which already "
                "points at a URL."
            )
    argv = resolve(descriptor, invocation)
    env = _env_for(descriptor)
    work_dir = (cwd or Path.cwd()).resolve()
    mounts = _plan_mounts(descriptor, invocation, work_dir)

    backend = _RUNTIMES[runtime]
    run_result: RunResult = backend.run(
        argv,
        container_image=descriptor.container_image,
        env=env,
        cwd=work_dir,
        mounts=mounts,
        runtime_args=runtime_args or [],
        stream=stream,
        capture=capture,
        image_path=image_path,
        no_pull=no_pull,
    )

    outputs = resolve_output_paths(descriptor, invocation, work_dir)
    outputs.extend(resolve_stdio_outputs(descriptor, run_result.stdout, run_result.stderr))
    return LaunchResult(
        command=argv,
        runtime=runtime,
        exit_code=run_result.exit_code,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        duration_seconds=run_result.duration_seconds,
        outputs=outputs,
    )


def _plan_mounts(
    descriptor: AnyDescriptor,
    invocation: dict[str, Any],
    work_dir: Path,
) -> list[Path]:
    """Combine the working directory with file-input parent dirs, deduped."""
    raw = {work_dir, *collect_file_mounts(descriptor, invocation)}
    # Re-dedupe the union so work_dir absorbs any nested file mounts.
    from boutiques.execution.mounts import _dedupe_descendants

    return _dedupe_descendants(raw)


def _env_for(descriptor: AnyDescriptor) -> dict[str, str]:
    """Build the environment dict declared by the descriptor."""
    env: dict[str, str] = {}
    for var in descriptor.environment_variables or []:
        env[var.name] = var.value
    return env


__all__ = ["LaunchResult", "launch"]
