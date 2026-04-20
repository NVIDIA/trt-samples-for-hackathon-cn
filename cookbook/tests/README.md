# Cookbook Test Runner Notes

This document describes how to configure and run cookbook example tests with the unified runner in `run_tests.py`.

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
  - 09-TRTLLM/**
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
- `--clean`: inject `TRT_COOKBOOK_CLEAN=1` and execute `clean`

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
