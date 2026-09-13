# 00 - Load Engine Plan

Load a TensorRT engine plan and its profiling data, then summarise the engine
structure and performance. This is the cookbook re-implementation of
`trt-engine-explorer`'s `EnginePlan` / `print_summary` / precision-stats API,
without pandas or plotly.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## What `main.py` shows

+ `case_load_and_summarize` - load the plan with `EnginePlan`, then `print_summary`
  (inputs/outputs, weight & activation footprint) and `print_precision_stats`
  (byte breakdown per precision).
+ `case_layer_report` - latency aggregated per layer type and the slowest individual layers.
+ `case_plot` - a two-panel Matplotlib figure (`overview.png`): latency-by-type
  bar chart + weights-by-precision pie chart, laid out with `GridSpec`.
