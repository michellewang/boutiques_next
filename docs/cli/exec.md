# `bosh exec`

Two subcommands: `simulate` (don't run) and `launch` (run).

## `simulate` {#simulate}

Resolve a descriptor + invocation into a shell-safe command-line string,
without running the tool.

```sh
bosh exec simulate <descriptor> <invocation>
```

```sh
$ bosh exec simulate fsl_bet.json invocation.json
bet /data/input.nii output.nii -f 0.5 -v
```

Internally, the template is tokenised once via `shlex.split`. Each
input's `value-key` is then replaced with a list of rendered tokens —
possibly zero (optional input omitted), one (a scalar), or many (a flag
+ value pair, a list with space separator). The output is rendered
through `shlex.join`, so values containing spaces are correctly quoted.

Before any tokenisation happens the invocation is validated against
the descriptor: structural shape via Pydantic (types, choices,
numeric ranges, list bounds), cross-input rules (`requires-inputs`,
`disables-inputs`, `value-requires`, `value-disables`), and group
constraints (`mutually-exclusive`, `one-is-required`, `all-or-none`).
Any failure prevents the simulate/launch and reports the violating
location.

```sh
$ bosh exec simulate descriptor.json invocation.json
tool 'value with spaces' --gpu=all
```

## `launch` {#launch}

Resolve and actually run the tool.

```sh
bosh exec launch <descriptor> <invocation>
                 [-r local|docker|singularity]
                 [--cwd DIR]
                 [--runtime-args "..."]
```

| Flag | Default | Description |
| --- | --- | --- |
| `-r`, `--runtime` | `local` | Runtime backend. |
| `--cwd` | `.` | Working directory for the run, also the container mount point. |
| `--runtime-args` | — | Extra arguments passed through to the container runtime, shlex-split. |

### Behaviour

- **`local`** — runs the resolved argv as a subprocess.
- **`docker`** — wraps in `docker run --rm -v <cwd>:<cwd> -w <cwd> <image> <argv>`. Auto-mounts the parent directory of every File-typed input in the invocation (deduped). Environment variables declared in the descriptor's `environment-variables` are passed via `-e`. The descriptor's `container-image.container-opts` are appended after our flags. `--no-pull` adds `--pull=never` so a locally-absent image is an error instead of an implicit pull.
- **`singularity`** — wraps in `singularity exec --bind <cwd> --pwd <cwd> <image>`. Docker images pull via the `docker://` URI scheme. Prefers `apptainer` if available. `--imagepath <file>` runs the local image file instead of a remote URI (singularity only); if the file does not exist it is pulled into place first (`<binary> pull <file> <uri>`). `--no-pull` refuses to run a remote URI — use `--imagepath` pointing at a pre-pulled file.

Container runtimes auto-mount the parent directory of every File-typed
input value present in the invocation, so input files are visible at
their host path inside the container. `--no-automounts` disables the file-input 
auto-mounts only; the working directory remains mounted.

Output streams to your terminal live; the final line is a `[bosh]`
summary plus the declared outputs and whether they appeared on disk:

```sh
$ bosh exec launch fsl_bet.json invocation.json -r docker
bet 1.0.0 (...)
Mask: out.nii.gz
...
[bosh] runtime=docker exit=0 duration=2.13s
[bosh] declared outputs:
  [OK] mask_file: /work/out.nii.gz
```

### Runtime arguments pass-through

Need a GPU? A specific network? Use `--runtime-args` — the string is
shlex-split and inserted into the container runtime invocation before
the image reference:

```sh
bosh exec launch desc.json inv.json -r docker --runtime-args "--gpus all --network host"
```

### Exit codes

The CLI exits with the tool's exit code on success/failure. A non-zero
exit from the underlying tool surfaces as-is; runtime errors (image
missing, descriptor without `container-image` for `-r docker`) exit
with code `2`.

## Python equivalent

```python
from boutiques.loader import load_descriptor
from boutiques.execution import simulate, launch

descriptor = load_descriptor("fsl_bet.json")
print(simulate(descriptor, invocation))            # str
result = launch(descriptor, invocation, runtime="docker")
print(result.exit_code, result.outputs)
```
