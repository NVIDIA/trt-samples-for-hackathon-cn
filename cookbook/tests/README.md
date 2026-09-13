# Cookbook Test Runner Notes

Configure and run cookbook example tests with the unified runner in `run_tests.py`.

## 1) File Location

Each example directory can optionally include a `unit_test.yaml`:

- `cookbook/02-API/Layer/Cast/unit_test.yaml`

If a directory does not contain `unit_test.yaml` but does contain `main.py`, the runner uses the default behavior:

- `run = ["python3 main.py > log-main.py.log"]`

Centralized directory skipping can be configured in `tests/skip_tests.yaml`:

```yaml
version: 1
skip:
  - 07-Tool/NetworkSerialization/TRT-8-version
```

Notes:

- `skip` supports both paths and glob patterns (same syntax as `--include/--exclude`)
- This file is intended for a global skip list; for temporary per-directory skipping, `.skip_unit_test` is still supported

## 2) Field Definitions (v1)

```yaml
version: 1                # Optional, default is 1
name: Cast / simple       # Optional, display name
enabled: true             # Optional, default is true

# Filter fields
tags: [api, layer, cast]  # Optional, array of strings

# Execution control
timeout: 1200             # Optional, timeout per command (seconds)
env:                       # Optional, inject environment variables
  MY_FLAG: "1"

# Lifecycle commands (string or array of strings)
pre:                       # Optional, execute before run
  - "make build"
run:                       # Required when using unit_test.yaml
  - "python3 main.py > log-main.py.log"
post:                      # Optional, execute after run
  - "python3 verify.py"

# Extra commands executed with --clean
clean:                     # Optional
  - "rm -rf *.log"
```

Constraints:

- `run` must be non-empty when `unit_test.yaml` exists
- `pre/run/post/clean` support:
  - a single string
  - an array of strings
- `env` values support scalars (`string/int/float/bool`) and are converted to strings

## 3) Recommended Migration Strategy

1. Introduce the runner first, without modifying example code
2. Add `unit_test.yaml` for “special directories” (for example, when explicit build steps or `main.sh` is needed)
3. For regular directories, no config is needed; use the default `main.py` rule
4. Finally, convert old `unit_test.sh` into a thin wrapper or remove it

## 4) CLI Arguments (`run_tests.py`)

Basics:

- Root directory is fixed to the `cookbook` path where the script is located (`ROOT = Path(__file__).resolve().parents[1]`)
- `--list`: list runnable examples only
- `--dry-run`: print commands only, do not execute
- `--summary-json PATH`: output a JSON summary report

Selection:

- `--case REL_PATH`: run one exact relative path (repeatable)
- `--include GLOB`: include by glob (repeatable, default `**`)
- `--exclude GLOB`: exclude by glob (repeatable)
- `--tags TAG`: run only examples containing any specified tag (repeatable)
- `--exclude-tags TAG`: exclude examples with specified tags (repeatable)

Execution:

- `--timeout SEC`: default timeout per command (default 1800)
- `--fail-fast`: stop on first failure
- `--clean`: execute `clean`

## 5) Configuration Examples

### 5.1 Regular Directory (Optional)

No `unit_test.yaml` is required; having `main.py` in the directory is enough.

### 5.2 Directory Requiring Extra Steps

```yaml
version: 1
tags: [plugin, compile]
pre:
  - "make build"
run:
  - "python3 main.py > log-main.py.log"
clean:
  - "make clean"
  - "rm -rf *.log"
```

### 5.3 Directory Without a `main.py` Entry

```yaml
version: 1
tags: [tool, polygraphy]
run:
  - "chmod +x main.sh"
  - "./main.sh"
  - "polygraphy run --help > Help-run.txt"
clean:
  - "rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt polygraphy_run.py"
```

## 6) Optional Packages

Some examples need a package that is **not** in the base environment. The policy, applied
consistently since 2026-09-08:

1. **The example degrades to a skip, never a crash.** Guard the import and exit 0:

   ```python
   try:
       import tensorrt_rtx as trt
   except ModuleNotFoundError:          # optional package, see this directory's README
       print("[SKIP] tensorrt_rtx is not installed (pip install tensorrt_rtx)")
       raise SystemExit(0)
   ```

   Catch `Exception`, not `ModuleNotFoundError`, when the package can *install* but fail to
   *import* — CuPy built for the wrong CUDA major version is the motivating case.

   For a **system** dependency (a binary, not a module) check the executable instead:

   ```python
   if shutil.which("dot") is None:
       print("[SKIP] the Graphviz `dot` binary is not on PATH (apt-get install graphviz)")
       raise SystemExit(0)
   ```

2. **The README says what is needed** and how to get it, in a block quote near the top.

3. **The `unit_test.yaml` carries the `optional-package` tag**, so the whole group can be
   re-tested after installing the extras:

   ```bash
   python3 tests/run_tests.py --tags optional-package
   ```

### Which packages, and why they are safe to add

| Package | Needed by | Note |
| ------- | --------- | ---- |
| `tensorrt_rtx` | `10-TensorRT-RTX/**`, `07-Tool/Polygraphy/TensorRTRTX` | separate Python module, does **not** shadow `tensorrt` |
| `tensorrt-lean`, `tensorrt-dispatch` | `04-Feature/LeanAndDispatchRuntime` | separate modules, same as above |
| `cupy` | `05-Plugin/PythonPlugin`, `05-Plugin/CuteDSLPlugin` | **carries its own CUDA runtime**, see below |
| Graphviz `dot` **binary** | `07-Tool/trex/11-ProcessEnginePipeline`, and `07-Tool/EngineVisualization` for the picture beside its `.gv` | `apt-get install graphviz`, **not** `pip` — the `graphviz` PyPI package only shells out to the binary. Installed here 2026-09-08 (2.43.0) |

### Why CuPy does not need a virtualenv

CuPy ships its own CUDA runtime, so the wheel must match the CUDA major version
(`cupy-cuda12x` / `cupy-cuda13x`). On this machine CuPy reports CUDA **12.9** while the system
toolkit is **13.3** — it works, using its bundled libraries. That is only tolerable because of two
properties, and **both must be preserved**:

+ `tensorrt_cookbook.utils_plugin` imports `cupy` **lazily, inside the function that uses it**, so
  `import tensorrt_cookbook` never pulls a second CUDA runtime into an unrelated example.
  (Verify with `python3 -c "import sys, tensorrt_cookbook; print('cupy' in sys.modules)"` → `False`.)
+ `run_tests.py` executes **every case in its own subprocess** (`subprocess.run(..., env=merged_env)`),
  so a CuPy build that is wrong for this machine can only break its own case.

Together these give the isolation a virtualenv would, without a second interpreter or a duplicated
`torch`. **When adding an example, import heavy optional packages inside the function — never at
module scope in a shared module.**

If a case ever genuinely needs a *different version* of a package that others also use, install it
side-by-side and point only that case at it — no virtualenv required, because `env:` is per-case:

```bash
pip install --target=/some/dir 'somepkg==1.2.3'
```

```yaml
env:
  PYTHONPATH: /some/dir
```
