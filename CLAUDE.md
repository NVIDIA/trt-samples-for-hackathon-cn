# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**The container is reset between sessions.** Everything *installed* is lost — the pip packages, the
apt packages, and `tensorrt_cookbook` itself — while the repository under `/work` survives. So the
first thing to do in a new session is put the environment back:

```bash
cd cookbook/
./setup_environment.sh
```

That installs the apt packages, the pip packages and `tensorrt_cookbook` (editable), served from
the download caches in `cookbook/dependence_packages/`. With the caches warm it takes about
**5 seconds**. It ends with a table of what is present, so a partial setup is visible rather than
assumed:

```txt
  [OK  ] the cookbook package
  [OK  ] TensorRT
  ...
  [OK  ] graphviz `dot` binary (apt)
  Everything is present
```

**There is no refresh step.** Both caches fill themselves on first use and are consulted
automatically afterwards; the script assumes the network is reachable, so a cache miss is slow,
never fatal. If something reports `MISS`, just run the script again.

The caches are **not** tracked by git — the whole `dependence_packages/` directory is ignored (see
`cookbook/dependence_packages.md`). They are a local accelerator and save download time only: a
fresh clone downloads everything once, and only later setups on that machine are served from the
cache.

**Recommended environment:** NVIDIA Docker image `nvcr.io/nvidia/pytorch:26.07-py3` (Python 3.12,
CUDA 13.3), with **TensorRT 11.3**. The image already provides torch, TensorRT, numpy and much else,
which is why a reinstall only fetches the ~25 packages that are actually missing rather than the
whole dependency tree.

**TensorRT 11.3 is the only supported version**, and code here may assume it. Workarounds for older
releases are removed rather than carried: the `use_patch_80` flag in
`utils_network_serialization.py`, which repaired the pre-11.3 "80/81" garbage rank for an unassigned
`trt.Dims`, was deleted when this rule came in. Note that removing a workaround is not the same as
removing the behaviour it dealt with - 11.3 still reports an unassigned `trt.Dims` as `None`, which
`add_slice()` rejects, so the *normalisation* stays while the version-compatibility code goes.

### Doing it by hand

```bash
cd cookbook/
export TRT_COOKBOOK_PATH=$(pwd)   # Optional: the package auto-discovers this when it can
pip install -r requirements.txt
pip install -e .                  # Editable install for development

# Release build:
python -m pip install -U build && python -m build && pip install dist/*.whl
```

`setup_environment.sh` points `PIP_CACHE_DIR` at `dependence_packages/pip` and passes
`--no-build-isolation` on the editable install, which reuses the installed `setuptools` instead of
fetching a fresh copy into a throwaway build environment (measured 2.9 s against 4.2 s).

## Common Commands

**Run a single example:**
```bash
cd 02-API/Layer/Cast
python3 main.py
```

**Run examples via the unified runner:**
```bash
python3 tests/run_tests.py                          # run all discoverable examples
python3 tests/run_tests.py --case 02-API/Layer/Cast # run one specific example
python3 tests/run_tests.py --include "02-API/**"    # glob filter
python3 tests/run_tests.py --list                   # list without running
python3 tests/run_tests.py --dry-run                # print commands only
python3 tests/run_tests.py --tags plugin            # filter by tag
python3 tests/run_tests.py --since HEAD             # only what your changes affect
python3 tests/run_tests.py --since HEAD --explain-since   # ... and why each file selected it
python3 tests/run_tests.py --framework tensorflow   # ONLY the TensorFlow examples
```

**Run the full test suite (unified runner):**
```bash
python3 tests/run_tests.py
```

**Run pytest tests (NetworkSerialization):**
```bash
pytest tests/NetworkSerialization/
pytest tests/NetworkSerialization/test_convolution.py  # single test file
```

**Linting / formatting:**
```bash
pre-commit run --all-files     # run all hooks (yapf, clang-format, autoflake, codespell…)
pre-commit install             # install hooks into .git/hooks
```

**Regenerate README.md:**
```bash
python3 build-README.py
```

**Add SPDX license headers to new files:**
```bash
python3 build-Copyright.py
```

## Code Architecture

### `tensorrt_cookbook/` — shared utility package

All examples import from this package. Key modules:

- **`utils_class.py`** — TensorRT wrapper classes that are the primary abstraction used across examples:
  - `TRTWrapperV1` / `TRTWrapperV2` — high-level build + inference workflows
  - `TRTWrapperDDS`, `TRTWrapperShapeInput`, `TRTWrapperV2Torch` — specialised variants
  - `CookbookLogger`, `CookbookProfiler`, `CookbookErrorRecorder` — diagnostic helpers
  - `CookbookGpuAllocator`, `CookbookOutputAllocator` — custom memory management
  - `FP16Tuning` — per-layer FP16/FP32 precision search, driven by `07-Tool/FP16Tuning/`. Its
    `tabulate`, `tqdm` and `modelopt` imports are **inside the methods**, not at module scope: this
    module is the first one `__init__.py` imports, and `import modelopt.onnx.autocast` alone costs
    3.6 s against 3.3 s for the whole package
- **`utils_cookbook.py`** — cookbook infrastructure: path resolution (`cookbook_path`), the `case_mark` decorator used by every `main.py`, logging, API-coverage inspection, README/copyright generation.
  `get_cookbook_logger` returns **one logger per log file**, named and non-propagating; it must not
  go back to configuring the root logger with `basicConfig`, which silently ignored every call after
  the first and swept third-party logging into the caller's file
- **`utils_function.py`** — framework-agnostic data helpers only: maths, array printing/comparison, dtype casting. Depends on nothing heavier than numpy/torch/tensorrt
- **`utils_network.py`** — building and inspecting `INetworkDefinition`: layer-type helpers (`layer_dynamic_cast`, …), `parse_onnx`, `print_network`, `export_network_as_onnx`
- **`utils_engine.py`** — inspecting a built engine: plan-file header (`print_engine_information`), engine/context I/O tables, engine-information JSON → ONNX (`export_engine_as_onnx`, and `export_engine_graph_as_onnx` which exports, runs `onnx.checker` and reports in one call)
- **`utils_workflow.py`** — `check_torch_operator`: run one Torch model through Torch → ONNX → ONNX-Runtime → Polygraphy → TensorRT and report which stage breaks
- **`utils_network_serialization.py`** — network serialization/deserialization (also tested by `tests/NetworkSerialization/`)
- **`utils_onnx.py`** — ONNX / onnx-graphsurgeon utilities. Deliberately does **not** import `tensorrt`
- **`utils_plugin.py`** — plugin development helpers
- **`utils_engine_explorer.py`** — TREx-derived engine profiling/visualisation
- **`engine_visualization.py`** — draws an engine layer-info JSON as Graphviz / Dagre HTML / yEd
  GraphML. Imports no `tensorrt`: a layer-info JSON is just a file. Defines `__all__`, because its
  label helpers (`short`, `sanitise`, …) are too generic for the package namespace and its
  `strip_layer_name_noise` would otherwise collide with the differently-behaved `clean_layer_name`
  in `utils_engine_explorer`

When adding a helper, place it by dependency direction, not by topic name: `utils_function` may not
import from the package, `utils_onnx` may not import `tensorrt`, and anything that touches a built
engine belongs in `utils_engine.py`. Because `__init__.py` re-exports every module with `import *`,
moving a function between these modules does not affect examples.

### Numbered example sections

| Directory            | Content                                                     |
| -------------------- | ----------------------------------------------------------- |
| `00-Data/`           | Dataset and model preparation                               |
| `01-SimpleDemo/`     | Minimal end-to-end TensorRT examples                        |
| `02-API/`            | TensorRT API coverage (Builder, Network, Layers, etc.)      |
| `03-Workflow/`       | Framework-to-TRT pipelines (PyTorch/TF/Paddle → ONNX → TRT) |
| `04-Feature/`        | Advanced features: quantization, profiling, caching, safety |
| `05-Plugin/`         | Custom plugin development and ONNX parser integration       |
| `06-DLFrameworkTRT/` | Torch-TensorRT                                              |
| `07-Tool/`           | External tools: trtexec, Polygraphy, Netron, ONNX utilities |
| `08-Advance/`        | Advanced patterns: CUDA graphs, multi-device, multi-stream  |
| `90-Misc/`           | General utilities not specific to TensorRT                  |

Each leaf directory is independently runnable. The standard entry point is `main.py`.

### Test orchestration

Examples are discovered and run by `tests/run_tests.py`. Discovery rules:
1. If a directory contains `unit_test.yaml`, its `run:` commands are used.
2. If a directory has `main.py` but no `unit_test.yaml`, the runner defaults to `python3 main.py > log-main.py.log`.
3. Paths matched by `tests/skip_tests.yaml` are globally excluded from discovery.
4. Presence of a `.skip_unit_test` file disables that directory entirely.

`unit_test.yaml` fields: `enabled`, `tags`, `timeout`, `env`, `pre`, `run`, `post`, `clean`. See `tests/run_tests.py` for the full spec.

**Every command that redirects to a log file must end in `2>&1`.** Without it stderr goes to the
terminal instead of the log, so a traceback found during a sweep is gone by the time anyone reads
the log file. The runner's own default command does this too.

**Two selectors that change what a default sweep means:**

+ **`--since REF`** runs only the examples your changes affect, which turns a ~40 minute sweep into
  seconds. A changed file is attributed to the example directory that owns it; a change under
  `tensorrt_cookbook/`, `include/`, `tests/` or `00-Data/`, or to a shared root file, selects
  everything; a `.md`/`.txt` selects nothing; and **a path no rule recognises selects everything**,
  because widening is the safe direction. `--explain-since` prints the reason per file.
+ **`--framework NAME`** is how the TensorFlow / JAX / PaddlePaddle examples run. They are skipped
  by default so a torch-only container does not watch them fail, and this flag is **exclusive**:
  it runs those and nothing else. The workflow is "default sweep, then `pip install tensorflow`,
  then `--framework tensorflow`". `--framework all` covers every gated framework.

## Code Style

- **Python**: formatted by YAPF (config in `.style.yapf`); isort is intentionally disabled to avoid conflicts.
- **C++/CUDA**: formatted by clang-format v16 (LLVM style, config in `.clang-format`).
- **CMake**: formatted by cmake-format.
- All source files require an SPDX Apache-2.0 header. Run `python3 copyright.py` to add/update headers.
- Pre-commit hooks enforce all of the above automatically on commit.
