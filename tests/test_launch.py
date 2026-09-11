"""Launch tests: local end-to-end + docker/singularity argv assertions."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from boutiques.execution import launch
from boutiques.execution.runtime.base import RunResult, RuntimeError_
from boutiques.loader import load_descriptor


def _echo_descriptor():
    """A trivial descriptor that wraps the platform 'echo'-like behavior."""
    return load_descriptor(
        {
            "schema-version": "0.5",
            "name": "echo_test",
            "description": "Print a message and exit",
            "tool-version": "1.0",
            "command-line": "[PYTHON] -c [SCRIPT] [MSG]",
            "inputs": [
                {
                    "id": "python",
                    "name": "P",
                    "type": "String",
                    "value-key": "[PYTHON]",
                },
                {
                    "id": "script",
                    "name": "S",
                    "type": "String",
                    "value-key": "[SCRIPT]",
                },
                {"id": "msg", "name": "M", "type": "String", "value-key": "[MSG]"},
            ],
        }
    )


def test_local_runtime_runs_real_subprocess(tmp_path):
    """End-to-end: invoke real Python via the local runtime."""
    descriptor = _echo_descriptor()
    result = launch(
        descriptor,
        {
            "python": sys.executable,
            "script": "import sys; print('hi from', sys.argv[1])",
            "msg": "boutiques",
        },
        runtime="local",
        cwd=tmp_path,
        stream=False,
    )
    assert result.exit_code == 0
    assert "hi from boutiques" in result.stdout
    assert result.runtime == "local"
    assert result.command[0] == sys.executable


def test_local_runtime_propagates_exit_code(tmp_path):
    descriptor = _echo_descriptor()
    result = launch(
        descriptor,
        {
            "python": sys.executable,
            "script": "import sys; sys.exit(7)",
            "msg": "",
        },
        runtime="local",
        cwd=tmp_path,
        stream=False,
    )
    assert result.exit_code == 7


def _docker_descriptor():
    return load_descriptor(
        {
            "schema-version": "0.5",
            "name": "in_container",
            "description": "Runs in a container",
            "tool-version": "1.0",
            "command-line": "do_thing [X]",
            "inputs": [
                {"id": "x", "name": "X", "type": "String", "value-key": "[X]"},
            ],
            "container-image": {"type": "docker", "image": "example/tool"},
        }
    )


def _fake_run_subprocess(captured: dict):
    """A run_subprocess stand-in that records its argv and returns success."""

    def _impl(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return RunResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    return _impl


def test_docker_runtime_wraps_argv(tmp_path):
    captured: dict = {}
    with patch(
        "boutiques.execution.runtime.docker.run_subprocess",
        side_effect=_fake_run_subprocess(captured),
    ):
        launch(_docker_descriptor(), {"x": "hello"}, runtime="docker", cwd=tmp_path)

    argv = captured["argv"]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-v" in argv and "-w" in argv
    assert "example/tool" in argv
    image_idx = argv.index("example/tool")
    assert argv[image_idx + 1 :] == ["do_thing", "hello"]


def test_docker_with_index_prepends_registry(tmp_path):
    descriptor = load_descriptor(
        {
            "schema-version": "0.5",
            "name": "x",
            "description": "y",
            "tool-version": "1.0",
            "command-line": "tool",
            "inputs": [
                {"id": "a", "name": "A", "type": "String", "value-key": "[A]"},
            ],
            "container-image": {
                "type": "docker",
                "image": "bids/mriqc",
                "index": "docker.io",
            },
        }
    )
    captured: dict = {}
    with patch(
        "boutiques.execution.runtime.docker.run_subprocess",
        side_effect=_fake_run_subprocess(captured),
    ):
        launch(descriptor, {"a": "val"}, runtime="docker", cwd=tmp_path)

    assert "docker.io/bids/mriqc" in captured["argv"]


def test_docker_requires_container_image(tmp_path):
    descriptor = _echo_descriptor()  # no container-image
    with pytest.raises(RuntimeError_, match="docker runtime requires"):
        launch(
            descriptor,
            {"python": "p", "script": "s", "msg": "m"},
            runtime="docker",
            cwd=tmp_path,
        )


def test_runtime_args_pass_through_to_docker(tmp_path):
    """--runtime-args content is inserted before the image ref, after our flags."""
    captured: dict = {}
    with patch(
        "boutiques.execution.runtime.docker.run_subprocess",
        side_effect=_fake_run_subprocess(captured),
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="docker",
            cwd=tmp_path,
            runtime_args=["--gpus", "all", "--network", "host"],
        )

    argv = captured["argv"]
    # All four runtime-arg tokens appear and are before the image ref.
    image_idx = argv.index("example/tool")
    for token in ("--gpus", "all", "--network", "host"):
        idx = argv.index(token)
        assert idx < image_idx, f"{token!r} should precede image ref"


def test_singularity_runtime_uses_docker_uri(tmp_path):
    captured: dict = {}
    with (
        patch(
            "boutiques.execution.runtime.singularity.run_subprocess",
            side_effect=_fake_run_subprocess(captured),
        ),
        patch(
            "boutiques.execution.runtime.singularity.shutil.which",
            return_value=None,
        ),
    ):
        launch(_docker_descriptor(), {"x": "hi"}, runtime="singularity", cwd=tmp_path)

    argv = captured["argv"]
    assert argv[0] == "singularity"
    assert "exec" in argv
    assert "--bind" in argv and "--pwd" in argv
    assert "docker://example/tool" in argv


def test_singularity_uses_existing_local_imagepath(tmp_path):
    img = tmp_path / "local.sif"
    img.write_bytes(b"")
    captured: dict = {}
    with (
        patch(
            "boutiques.execution.runtime.singularity.run_subprocess",
            side_effect=_fake_run_subprocess(captured),
        ),
        patch(
            "boutiques.execution.runtime.singularity.shutil.which",
            return_value=None,
        ),
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="singularity",
            cwd=tmp_path,
            image_path=img,
        )

    argv = captured["argv"]
    assert argv[0] == "singularity"
    assert str(img) in argv
    assert not any("docker://" in token for token in argv)


def test_cli_launch_imagepath_uses_local_image(tmp_path):
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
                "container-image": {"type": "docker", "image": "example/tool"},
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "v"}')
    img = tmp_path / "local.sif"
    img.write_bytes(b"")

    captured: dict = {}
    with (
        patch(
            "boutiques.execution.runtime.singularity.run_subprocess",
            side_effect=_fake_run_subprocess(captured),
        ),
        patch(
            "boutiques.execution.runtime.singularity.shutil.which",
            return_value=None,
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "exec",
                "launch",
                str(descriptor_path),
                str(inv_path),
                "-r",
                "singularity",
                "--imagepath",
                str(img),
            ],
        )

    assert result.exit_code == 0
    argv = captured["argv"]
    assert str(img) in argv
    assert not any("docker://" in token for token in argv)


def test_singularity_auto_pulls_missing_imagepath(tmp_path):
    img = tmp_path / "pulled.sif"
    pull_argv: list[list[str]] = []
    exec_captured: dict = {}

    def fake_run(argv, **kwargs):
        if argv[1] == "pull":
            pull_argv.append(argv)
            img.write_bytes(b"")
            return RunResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)
        exec_captured["argv"] = argv
        return RunResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    with (
        patch(
            "boutiques.execution.runtime.singularity.run_subprocess",
            side_effect=fake_run,
        ),
        patch(
            "boutiques.execution.runtime.singularity.shutil.which",
            return_value=None,
        ),
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="singularity",
            cwd=tmp_path,
            image_path=img,
        )

    assert pull_argv == [["singularity", "pull", str(img), "docker://example/tool"]]
    assert str(img) in exec_captured["argv"]


def test_singularity_no_pull_missing_imagepath_errors(tmp_path):
    img = tmp_path / "missing.sif"
    with pytest.raises(RuntimeError_, match="--no-pull"):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="singularity",
            cwd=tmp_path,
            image_path=img,
            no_pull=True,
        )


def test_singularity_no_pull_allows_existing_imagepath(tmp_path):
    img = tmp_path / "local.sif"
    img.write_bytes(b"")
    captured: dict = {}
    with (
        patch(
            "boutiques.execution.runtime.singularity.run_subprocess",
            side_effect=_fake_run_subprocess(captured),
        ),
        patch(
            "boutiques.execution.runtime.singularity.shutil.which",
            return_value=None,
        ),
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="singularity",
            cwd=tmp_path,
            image_path=img,
            no_pull=True,
        )

    assert str(img) in captured["argv"]


def test_singularity_no_pull_refuses_remote_uri(tmp_path):
    with pytest.raises(
        RuntimeError_, match="--no-pull is specified without --imagepath"
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="singularity",
            cwd=tmp_path,
            no_pull=True,
        )


def test_imagepath_rejected_for_non_singularity_runtime(tmp_path):
    img = tmp_path / "local.sif"
    img.write_bytes(b"")
    for runtime in ("docker", "local"):
        with pytest.raises(
            RuntimeError_, match="only applies to the singularity runtime"
        ):
            launch(
                _docker_descriptor(),
                {"x": "hi"},
                runtime=runtime,
                cwd=tmp_path,
                image_path=img,
            )


def test_cli_launch_imagepath_with_docker_errors(tmp_path):
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
                "container-image": {"type": "docker", "image": "example/tool"},
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "v"}')
    img = tmp_path / "local.sif"
    img.write_bytes(b"")

    result = CliRunner().invoke(
        app,
        [
            "exec",
            "launch",
            str(descriptor_path),
            str(inv_path),
            "-r",
            "docker",
            "--imagepath",
            str(img),
        ],
    )
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "only applies to the singularity runtime" in combined


def test_docker_no_pull_adds_pull_never(tmp_path):
    captured: dict = {}
    with patch(
        "boutiques.execution.runtime.docker.run_subprocess",
        side_effect=_fake_run_subprocess(captured),
    ):
        launch(
            _docker_descriptor(),
            {"x": "hi"},
            runtime="docker",
            cwd=tmp_path,
            no_pull=True,
        )

    assert "--pull=never" in captured["argv"]


def test_cli_force_docker_aliases_runtime(tmp_path):
    """`--force-docker` (classic-bosh compat) selects the docker runtime, -v appends mounts."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
                "container-image": {"type": "docker", "image": "example/tool"},
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "v"}')

    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return RunResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    with patch(
        "boutiques.execution.runtime.docker.run_subprocess", side_effect=fake_run
    ):
        result = CliRunner().invoke(
            app,
            [
                "exec",
                "launch",
                str(descriptor_path),
                str(inv_path),
                "--force-docker",
                "-v",
                "/a:/b",
                "-v",
                "/c:/d",
            ],
        )

    assert result.exit_code == 0
    argv = captured["argv"]
    assert argv[0] == "docker"
    for pair in ("/a:/b", "/c:/d"):
        assert pair in argv


def test_cli_force_multiple_runtimes_errors(tmp_path):
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
                "container-image": {"type": "docker", "image": "example/tool"},
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "v"}')

    result = CliRunner().invoke(
        app,
        [
            "exec",
            "launch",
            str(descriptor_path),
            str(inv_path),
            "--force-docker",
            "--force-singularity",
        ],
    )
    assert result.exit_code == 1
    combined = (result.stdout or "") + (result.stderr or "")
    assert "at most one" in combined


def test_cli_simulate_accepts_i_flag(tmp_path):
    """Classic-bosh -i should provide the invocation as an alternative to positional."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "hello"}')

    result = CliRunner().invoke(
        app, ["exec", "simulate", str(descriptor_path), "-i", str(inv_path)]
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_cli_simulate_rejects_positional_invocation(tmp_path):
    """Classic simulate takes the invocation via -i only; no positional form."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
            }
        )
    )
    inv_path = tmp_path / "inv.json"
    inv_path.write_text('{"x": "v"}')

    result = CliRunner().invoke(
        app,
        ["exec", "simulate", str(descriptor_path), str(inv_path)],
    )
    assert result.exit_code != 0


def test_cli_simulate_accepts_invocation_as_json_string(tmp_path):
    """Classic -i accepts a JSON string, not just a file path."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
            }
        )
    )

    result = CliRunner().invoke(
        app, ["exec", "simulate", str(descriptor_path), "-i", '{"x": "hello"}']
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_cli_simulate_includes_descriptor_defaults(tmp_path):
    """Optional inputs with a default-value appear in the simulated command."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X] [SPECIES]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"},
                    {
                        "id": "species",
                        "name": "Species",
                        "type": "String",
                        "optional": True,
                        "default-value": "human",
                        "command-line-flag": "--species",
                        "value-key": "[SPECIES]",
                    },
                ],
            }
        )
    )

    result = CliRunner().invoke(
        app, ["exec", "simulate", str(descriptor_path), "-i", '{"x": "v"}']
    )
    assert result.exit_code == 0
    assert "--species human" in result.stdout


def test_default_value_satisfies_validation_like_classic():
    """Classic fills default-values before validating; an omitted default that
    satisfies a one-is-required group must not be rejected."""
    from boutiques.invocation_check import validate_invocation

    descriptor = load_descriptor(
        {
            "schema-version": "0.5",
            "name": "t",
            "description": "x",
            "tool-version": "1.0",
            "command-line": "tool [A] [B]",
            "inputs": [
                {
                    "id": "a",
                    "name": "A",
                    "type": "String",
                    "optional": True,
                    "default-value": "on",
                    "value-key": "[A]",
                },
                {
                    "id": "b",
                    "name": "B",
                    "type": "String",
                    "optional": True,
                    "value-key": "[B]",
                },
            ],
            "groups": [
                {
                    "id": "g",
                    "name": "G",
                    "members": ["a", "b"],
                    "one-is-required": True,
                },
            ],
        }
    )
    # Neither member supplied, but 'a' has a default -> classic accepts it.
    assert validate_invocation(descriptor, {}) == []


def test_cli_launch_unimplemented_flag_refuses(tmp_path):
    """A not-yet-implemented classic flag exits non-zero with an issue pointer."""
    import json

    from typer.testing import CliRunner

    from boutiques.cli import app

    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "schema-version": "0.5",
                "name": "t",
                "description": "x",
                "tool-version": "1.0",
                "command-line": "tool [X]",
                "inputs": [
                    {"id": "x", "name": "X", "type": "String", "value-key": "[X]"}
                ],
            }
        )
    )

    result = CliRunner().invoke(
        app,
        ["exec", "launch", str(descriptor_path), '{"x": "v"}', "--user"],
    )
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "not implemented" in combined
    assert "issues" in combined


def test_unknown_runtime_raises():
    descriptor = _echo_descriptor()
    with pytest.raises(RuntimeError_, match="Unknown runtime"):
        launch(
            descriptor,
            {"python": "p", "script": "s", "msg": "m"},
            runtime="podman",
        )


def test_output_paths_resolve_against_cwd(tmp_path):
    """Path-template substitution + existence check (no subprocess)."""
    descriptor = load_descriptor(
        {
            "schema-version": "0.5",
            "name": "with_output",
            "description": "x",
            "tool-version": "1.0",
            "command-line": "true [NAME]",
            "inputs": [
                {"id": "name", "name": "N", "type": "String", "value-key": "[NAME]"},
            ],
            "output-files": [
                {"id": "out", "name": "Out", "path-template": "[NAME].txt"}
            ],
        }
    )
    (tmp_path / "result.txt").write_text("data")

    from boutiques.execution.outputs import resolve_output_paths

    outputs = resolve_output_paths(descriptor, {"name": "result"}, tmp_path)
    assert len(outputs) == 1
    assert outputs[0].path == (tmp_path / "result.txt").resolve()
    assert outputs[0].exists


def test_environment_variables_are_passed(tmp_path):
    """Env vars declared by the descriptor reach the subprocess."""
    descriptor = load_descriptor(
        {
            "schema-version": "0.5",
            "name": "env_check",
            "description": "x",
            "tool-version": "1.0",
            "command-line": "[PYTHON] -c [SCRIPT]",
            "inputs": [
                {
                    "id": "python",
                    "name": "P",
                    "type": "String",
                    "value-key": "[PYTHON]",
                },
                {
                    "id": "script",
                    "name": "S",
                    "type": "String",
                    "value-key": "[SCRIPT]",
                },
            ],
            "environment-variables": [
                {"name": "BOUTIQUES_TEST_VAR", "value": "from_descriptor"}
            ],
        }
    )
    result = launch(
        descriptor,
        {
            "python": sys.executable,
            "script": "import os; print(os.environ['BOUTIQUES_TEST_VAR'])",
        },
        runtime="local",
        cwd=tmp_path,
        stream=False,
    )
    assert result.exit_code == 0
    assert "from_descriptor" in result.stdout
