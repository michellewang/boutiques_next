"""Singularity / Apptainer runtime."""

from __future__ import annotations

import shutil
from pathlib import Path

from boutiques.execution.runtime._subprocess import run_subprocess
from boutiques.execution.runtime.base import RunResult, RuntimeError_
from boutiques.models.v05.containers import DockerOrSingularityImage, RootfsImage


def run(
    argv: list[str],
    *,
    container_image: object | None,
    env: dict[str, str],
    cwd: Path,
    mounts: list[Path] | None = None,
    runtime_args: list[str] | None = None,
    stream: bool = True,
    capture: bool = True,
    image_path: Path | None = None,
    no_pull: bool = False,
) -> RunResult:
    if container_image is None:
        raise RuntimeError_(
            "singularity runtime requires the descriptor to declare a container-image."
        )
    binary = _resolve_binary()
    image_arg = _resolve_image_arg(binary, container_image, image_path, no_pull)
    cwd_abs = cwd.resolve()

    wrapper: list[str] = [binary, "exec"]
    for mount in mounts or [cwd_abs]:
        wrapper.extend(["--bind", str(mount)])
    wrapper.extend(["--pwd", str(cwd_abs)])
    for k, v in env.items():
        wrapper.extend(["--env", f"{k}={v}"])
    if runtime_args:
        wrapper.extend(runtime_args)
    wrapper.append(image_arg)
    wrapper.extend(argv)

    return run_subprocess(wrapper, stream=stream, capture=capture)


def _resolve_image_arg(
    binary: str,
    container_image: object,
    image_path: Path | None,
    no_pull: bool,
) -> str:
    image_uri = _image_uri(container_image)
    if image_path is not None:
        path = image_path.expanduser().resolve()
        return _ensure_image(binary, path, image_uri, no_pull)
    if no_pull:
        raise RuntimeError_(
            f"--no-pull cannot run the remote image {image_uri!r}: the runtime would pull "
            "it into its own cache on first use. Use --imagepath to point at a "
            "pre-pulled local image file."
        )
    return image_uri


def _ensure_image(binary: str, path: Path, pull_uri: str, no_pull: bool) -> str:
    if path.exists():
        return str(path)
    if no_pull:
        raise RuntimeError_(
            f"Image not found at {path} and --no-pull is set. Pull it first, e.g. "
            f"{binary} pull {path} {pull_uri}"
        )
    if not path.parent.exists():
        raise RuntimeError_(
            f"Cannot pull image: parent directory {path.parent} does not exist."
        )
    pull = run_subprocess([binary, "pull", str(path), pull_uri])
    if pull.exit_code != 0:
        raise RuntimeError_(
            f"Failed to pull container image to {path} (exit {pull.exit_code}):\n"
            f"{pull.stderr}"
        )
    if not path.exists():
        raise RuntimeError_(
            f"Pull succeeded but created no image at {path}:\n{pull.stdout}\n{pull.stderr}"
        )
    return str(path)


def _resolve_binary() -> str:
    if shutil.which("apptainer"):
        return "apptainer"
    if shutil.which("singularity"):
        return "singularity"
    return "singularity"  # let subprocess surface a clean FileNotFoundError


def _image_uri(container_image: object) -> str:
    if isinstance(container_image, DockerOrSingularityImage):
        if container_image.index:
            return f"docker://{container_image.index}/{container_image.image}"
        return f"docker://{container_image.image}"
    if isinstance(container_image, RootfsImage):
        return str(container_image.url)
    raise RuntimeError_(
        f"Unsupported container-image type: {type(container_image).__name__}"
    )
