# 01 - Report Card: Latency

A latency "report card" that shows where a TensorRT engine spends its time.
This is the cookbook re-implementation of `trt-engine-explorer`'s
`report_card_perf_overview` / `layer_latency_sunburst` / `plot_engine_timings`.

The original renders an **interactive plotly dropdown** of ~10 views; here each
view is a Matplotlib figure **saved to a PNG file**. Range-Sliders and hover
tooltips are dropped; instead the plotting range/size is controlled by the
`n_top_layers` / `hist_bins` knobs at the top of `main.py` (change and re-run).

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## Figures produced

| File                             | Content                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `latency_by_type.png`            | latency (ms and %) and layer count, aggregated per layer type  |
| `latency_per_layer.png`          | the slowest `n_top_layers` layers, colored by layer type       |
| `latency_distribution.png`       | histogram of the per-layer latency-% contribution              |
| `precision_rollup.png`           | layer count and latency share grouped by precision (pie)       |
| `latency_by_type_precision.png`  | stacked bar of latency per type split by precision (2D replacement for the type/precision sunburst) |
| `engine_timings.png`             | per-iteration end-to-end latency samples from the timing JSON  |
