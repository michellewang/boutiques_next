"""`bosh exec` — simulate or launch a descriptor with an invocation.

The flag surface mirrors classic ``bosh exec`` so existing tooling keeps
working. Classic options this runtime does not honor yet are still accepted
by the parser but refuse loudly (see :mod:`boutiques.cli._compat`) instead of
being silently ignored.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import typer

from boutiques.cli._compat import not_implemented
from boutiques.cli._input import load_descriptor_or_exit, load_invocation_or_exit
from boutiques.execution import launch as _launch
from boutiques.execution import simulate as _simulate
from boutiques.execution.runtime.base import RuntimeError_
from boutiques.invocation_check import InvocationValidationError

exec_app = typer.Typer(
    name="exec",
    help="Simulate or launch a descriptor.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@exec_app.command("simulate")
def simulate(
    descriptor: str = typer.Argument(
        ..., help="Path, http(s) URL, or JSON string of a Boutiques descriptor."
    ),
    input_: str | None = typer.Option(
        None,
        "-i",
        "--input",
        help="Invocation as a JSON file path or JSON string.",
    ),
    complete: bool = typer.Option(  # accepted for classic compat; currently a no-op
        False,
        "-c",
        "--complete",
        help="Classic compat: include optional parameters (defaults are always included).",
    ),
    json_output: bool = typer.Option(
        False,
        "-j",
        "--json",
        help="Classic compat: emit the completed invocation as JSON (not implemented yet).",
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        help="Classic compat: fetch the descriptor from Zenodo's sandbox (not implemented yet).",
    ),
) -> None:
    """Resolve a descriptor + invocation into a command-line without running it."""
    if json_output:
        not_implemented("--json")
    if sandbox:
        not_implemented("--sandbox")

    parsed = load_descriptor_or_exit(descriptor)

    if input_ is None:
        typer.echo(
            "Provide an invocation via --input/-i. "
            "To generate an example invocation, use `bosh example`.",
            err=True,
        )
        raise typer.Exit(1)

    inv = load_invocation_or_exit(input_)
    try:
        typer.echo(_simulate(parsed, inv))
    except InvocationValidationError as exc:
        typer.echo(f"Invocation invalid:\n{exc}", err=True)
        raise typer.Exit(1) from exc


@exec_app.command("launch")
def launch(
    descriptor: str = typer.Argument(
        ..., help="Path, http(s) URL, or JSON string of a Boutiques descriptor."
    ),
    invocation: str = typer.Argument(
        ..., help="Invocation as a JSON file path or JSON string."
    ),
    volumes: list[str] = typer.Option(
        [],
        "-v",
        "--volumes",
        help="HOST:CONTAINER bind mount, passed to the container runtime. Repeatable.",
    ),
    container_opts: list[str] = typer.Option(
        [],
        "--container-opts",
        help=(
            "Extra arguments passed through to the container runtime; each value is "
            "shlex-split. Repeatable. Example: --container-opts '--gpus all'."
        ),
    ),
    runtime: str = typer.Option(
        "local",
        "-r",
        "--runtime",
        help="Runtime backend: local, docker, singularity.",
    ),
    cwd: Path = typer.Option(
        Path.cwd(),
        "--cwd",
        help="Working directory for the run (also the container mount point).",
    ),
    no_container: bool = typer.Option(
        False,
        "--no-container",
        help="Run on the host with no container (equivalent to -r local).",
    ),
    stream: bool = typer.Option(  # always-on in this runtime; accepted for compat
        False,
        "-s",
        "--stream",
        help="Classic compat: stream stdout/stderr in real time (always on here).",
    ),
    debug: bool = typer.Option(  # accepted for classic compat; currently a no-op
        False,
        "-x",
        "--debug",
        help="Classic compat: accepted, currently a no-op.",
    ),
    skip_data_collection: bool = typer.Option(  # no data collection here; accepted for compat
        False,
        "--skip-data-collection",
        help="Classic compat: this runtime never collects execution data.",
    ),
    force_docker: bool = typer.Option(
        False, "--force-docker", help="Compat: equivalent to '-r docker'."
    ),
    force_singularity: bool = typer.Option(
        False,
        "--force-singularity",
        help="Compat: equivalent to '-r singularity'.",
    ),
    force_apptainer: bool = typer.Option(
        False,
        "--force-apptainer",
        help="Compat: equivalent to '-r singularity' (Apptainer is selected automatically).",
    ),
    imagepath: str | None = typer.Option(
        None,
        "--imagepath",
        help=(
            "Path to a local container image (singularity only; pulled into place if missing)."
        ),
    ),
    user: bool = typer.Option(
        False,
        "-u",
        "--user",
        help="Classic compat: run the container as the local user (not implemented yet).",
    ),
    provenance: str | None = typer.Option(
        None,
        "--provenance",
        help="Classic compat: append JSON to the execution record (not implemented yet).",
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        help="Classic compat: fetch the descriptor from Zenodo's sandbox (not implemented yet).",
    ),
    no_pull: bool = typer.Option(
        False,
        "--no-pull",
        help=(
            "Do not pull the container image: docker passes --pull=never; "
            "singularity refuses remote images unless --imagepath points at a local file."
        ),
    ),
    no_automounts: bool = typer.Option(
        False,
        "--no-automounts",
        help="Classic compat: disable auto-mounting input files (not implemented yet).",
    ),
) -> None:
    """Launch a descriptor with an invocation under the chosen runtime."""
    if user:
        not_implemented("--user")
    if provenance is not None:
        not_implemented("--provenance")
    if sandbox:
        not_implemented("--sandbox")
    if no_automounts:
        not_implemented("--no-automounts")

    parsed = load_descriptor_or_exit(descriptor)

    try:
        runtime = _resolve_runtime(
            runtime, force_docker, force_singularity, force_apptainer, no_container
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    extra_args: list[str] = []
    for opt in container_opts:
        extra_args.extend(shlex.split(opt))
    for vol in volumes:
        extra_args.extend(["-v", vol])

    inv = load_invocation_or_exit(invocation)
    try:
        result = _launch(
            parsed,
            inv,
            runtime=runtime,
            cwd=cwd,
            runtime_args=extra_args,
            stream=True,
            capture=False,  # output already streamed; don't buffer twice
            image_path=Path(imagepath).resolve() if imagepath else None,
            no_pull=no_pull,
        )
    except InvocationValidationError as exc:
        typer.echo(f"Invocation invalid:\n{exc}", err=True)
        raise typer.Exit(1) from exc
    except RuntimeError_ as exc:
        typer.echo(f"Runtime error: {exc}", err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"\n[bosh] command: {shlex.join(result.command)}")
    typer.echo(
        f"[bosh] runtime={result.runtime} "
        f"exit={result.exit_code} "
        f"duration={result.duration_seconds:.2f}s"
    )
    if result.outputs:
        typer.echo("[bosh] declared outputs:")
        for o in result.outputs:
            marker = "OK" if o.exists else "missing"
            typer.echo(f"  [{marker}] {o.id}: {o.path}")
    raise typer.Exit(result.exit_code)


def _resolve_runtime(
    runtime: str,
    force_docker: bool,
    force_singularity: bool,
    force_apptainer: bool,
    no_container: bool,
) -> str:
    """Reconcile -r with classic bosh's --force-* / --no-container compat flags."""
    selected: set[str] = set()
    if force_docker:
        selected.add("docker")
    if force_singularity or force_apptainer:
        selected.add("singularity")
    if no_container:
        selected.add("local")

    if len(selected) > 1:
        raise ValueError(
            "Pass at most one of --no-container, --force-docker, "
            "--force-singularity, --force-apptainer."
        )
    if selected:
        forced = next(iter(selected))
        # If -r was also explicitly set to a conflicting non-default value, error.
        if runtime != "local" and runtime != forced:
            raise ValueError(
                f"--force-*/--no-container selects {forced!r} but -r is set to "
                f"{runtime!r}; pick one."
            )
        return forced
    return runtime


def register(app: typer.Typer) -> None:
    app.add_typer(exec_app)
