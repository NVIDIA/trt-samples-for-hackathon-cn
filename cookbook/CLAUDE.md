# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Mandatory before running any example:**
```bash
export TRT_COOKBOOK_PATH=$(pwd)   # must point to the cookbook/ root
```

The `tensorrt_cookbook` package validates this at import time and calls `sys.exit(1)` if it is missing or invalid.

**Recommended environment:** NVIDIA Docker image `nvcr.io/nvidia/pytorch:25.10-py3` (Python 3.12, CUDA 13.0, TensorRT 10.13).

**Install dependencies:**
```bash
pip install -r requirements.txt

# Developer (editable) install:
pip install -e .

# Release build:
python -m pip install -U build && python -m build && pip install dist/*.whl
```

## Common Commands

**Run a single example:**
```bash
cd 02-API/Layer/Cast
python3 main.py
```

**Run examples via the unified runner:**
```bash
python3 tools/run_examples.py                          # run all discoverable examples
python3 tools/run_examples.py --case 02-API/Layer/Cast # run one specific example
python3 tools/run_examples.py --include "02-API/**"    # glob filter
python3 tools/run_examples.py --list                   # list without running
python3 tools/run_examples.py --dry-run                # print commands only
python3 tools/run_examples.py --tags plugin            # filter by tag
```

**Run the full test suite (shell-based):**
```bash
./unit_test.sh
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
python3 copyright.py
```

## Code Architecture

### `tensorrt_cookbook/` — shared utility package

All examples import from this package. Key modules:

- **`utils_class.py`** — TensorRT wrapper classes that are the primary abstraction used across examples:
  - `TRTWrapperV1` / `TRTWrapperV2` — high-level build + inference workflows
  - `TRTWrapperDDS`, `TRTWrapperShapeInput`, `TRTWrapperV2Torch` — specialised variants
  - `CookbookLogger`, `CookbookProfiler`, `CookbookErrorRecorder` — diagnostic helpers
  - `CookbookGpuAllocator`, `CookbookOutputAllocator` — custom memory management
- **`utils_function.py`** — dtype casting, array printing, CUDA utilities
- **`utils_network.py`** — network building helpers
- **`utils_network_serialization.py`** — network serialization/deserialization (also tested by `tests/NetworkSerialization/`)
- **`utils_onnx.py`** — ONNX graph utilities
- **`utils_plugin.py`** — plugin development helpers

### Numbered example sections

| Directory | Content |
|-----------|---------|
| `00-Data/` | Dataset and model preparation |
| `01-SimpleDemo/` | Minimal end-to-end TensorRT examples |
| `02-API/` | TensorRT API coverage (Builder, Network, Layers, etc.) |
| `03-Workflow/` | Framework-to-TRT pipelines (PyTorch/TF/Paddle → ONNX → TRT) |
| `04-Feature/` | Advanced features: quantization, profiling, caching, safety |
| `05-Plugin/` | Custom plugin development and ONNX parser integration |
| `06-DLFrameworkTRT/` | Torch-TensorRT |
| `07-Tool/` | External tools: trtexec, Polygraphy, Netron, ONNX utilities |
| `08-Advance/` | Advanced patterns: CUDA graphs, multi-device, multi-stream |
| `09-TRTLLM/` | TensorRT-LLM tools |
| `98-Uncategorized/` | General utilities not specific to TensorRT |

Each leaf directory is independently runnable. The standard entry point is `main.py`.

### Test orchestration

Examples are discovered and run by `tools/run_examples.py`. Discovery rules:
1. If a directory contains `unit_test.yaml`, its `run:` commands are used.
2. If a directory has `main.py` but no `unit_test.yaml`, the runner defaults to `python3 main.py > log-main.py.log`.
3. Presence of a `.skip_unit_test` file disables that directory entirely.

`unit_test.yaml` fields: `enabled`, `tags`, `timeout`, `env`, `pre`, `run`, `post`, `clean`. See `tools/example-spec.md` for the full spec.

## Code Style

- **Python**: formatted by YAPF (config in `.style.yapf`); isort is intentionally disabled to avoid conflicts.
- **C++/CUDA**: formatted by clang-format v16 (LLVM style, config in `.clang-format`).
- **CMake**: formatted by cmake-format.
- All source files require an SPDX Apache-2.0 header. Run `python3 copyright.py` to add/update headers.
- Pre-commit hooks enforce all of the above automatically on commit.
