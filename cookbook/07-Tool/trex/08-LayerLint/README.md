# 08 - Layer Lint

Run heuristic **linters** over an engine to flag potential performance hazards,
in the spirit of `trt-engine-explorer`'s `lint.py` (a preview feature there).
The linters operate on the engine-graph JSON only - no GPU or profiling data
required.

Ported linters (`tensorrt_cookbook/utils_engine_explorer.py`):

+ `lint_convolutions` - convolutions not accelerated by Tensor Cores, quantized
  convolutions with float outputs, and non-optimal channel alignment.
+ `lint_reformats` - Reformat layers that convert between data types (not just layouts).
+ `lint_slices` - Slice layers that convert between data types.
+ `lint_qdq` - dangling / unfused Quantize / Dequantize (Scale) layers.

`lint_engine(plan)` runs all of them and returns `{category: [hazards]}`.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # run the linters (no GPU required)
```

## Output

+ `case_lint_report` - text list of every hazard (name, hazard, mitigation, help, ...).
+ `lint_summary.png` - bar chart of the hazard count per linter category.

Note: these are heuristics/preview checks; a flagged layer is not necessarily a
real problem, and TensorRT may already be optimal.
