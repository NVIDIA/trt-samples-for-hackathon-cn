# 06 - Report Card: Precision

Inspect how precisions (INT8 / FP16 / FP32 / ...) are used across the engine and
where TensorRT inserts **Reformat** layers to convert data. This is the cookbook
re-implementation of the precision-related views from `trt-engine-explorer`'s
`report_card_perf_overview` (precision per layer / precision per type sunburst /
precision statistics) plus `report_card_reformat_overview`.

The original renders interactive plotly dropdowns; here each view is a Matplotlib
figure saved to a PNG file (`n_top_layers` knob at the top of `main.py`).

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## Output

+ `case_precision_stats` - text byte breakdown per precision (activations and weights).
+ `precision_per_layer.png` - per-layer latency, colored by the layer's precision.
+ `precision_stats.png` - bytes per precision for input activations / output activations / weights.
+ `precision_per_type.png` - layer count per type split by precision (2D replacement for the precision sunburst).
+ `reformat_overview.png` - Reformat layer count and % latency grouped by origin (e.g. `QDQ`).
