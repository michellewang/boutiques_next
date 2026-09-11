"""Docker runtime: wrap the argv in a ``docker run`` invocation."""

from __future__ import annotations

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
    image_path: Path | None = None,  # rejected in launch(); accepted for a uniform call
    no_pull: bool = False,
) -> RunResult:
    if container_image is None:
        raise RuntimeError_("docker runtime requires the descriptor to declare a container-image.")
    image_ref = _image_ref(container_image)
    cwd_abs = cwd.resolve()

    wrapper: list[str] = ["docker", "run", "--rm"]
    if no_pull:
        wrapper.append("--pull=never")
    for mount in mounts or [cwd_abs]:
        wrapper.extend(["-v", f"{mount}:{mount}"])
    wrapper.extend(["-w", str(cwd_abs)])
    for k, v in env.items():
        wrapper.extend(["-e", f"{k}={v}"])
    if runtime_args:
        wrapper.extend(runtime_args)
    if isinstance(container_image, DockerOrSingularityImage) and container_image.container_opts:
        wrapper.extend(container_image.container_opts)
    wrapper.append(image_ref)
    wrapper.extend(argv)

    return run_subprocess(wrapper, stream=stream, capture=capture)


def _image_ref(container_image: object) -> str:
    if isinstance(container_image, DockerOrSingularityImage):
        if container_image.index:
            return f"{container_image.index}/{container_image.image}"
        return container_image.image
    if isinstance(container_image, RootfsImage):
        raise RuntimeError_(
            "docker runtime cannot launch a rootfs-type container image. "
            "Use the singularity runtime instead."
        )
    raise RuntimeError_(f"Unsupported container-image type: {type(container_image).__name__}")
