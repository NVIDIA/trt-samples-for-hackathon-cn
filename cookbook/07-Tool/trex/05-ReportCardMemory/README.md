# 05 - Report Card: Memory

Inspect where the engine spends **memory** - weights vs activations - in the
spirit of `trt-engine-explorer`'s `report_card_memory_footprint`.

Per-layer byte columns (computed by `EnginePlan`):

+ `weights_size` - the layer's constant weights
+ `total_io_size_bytes` - the layer's input + output activations
+ `total_footprint_bytes` - weights + activations

The original renders an interactive plotly dropdown; here each view is a
Matplotlib figure saved to a PNG file (`n_top_layers` / `hist_bins` knobs at the
top of `main.py`).

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```

## Output

+ `case_memory_table` - total weights / activations / footprint and the largest-footprint layers.
+ `footprint_per_layer.png` - stacked bar (weights + activations) for the top `n_top_layers` layers.
+ `footprint_by_type.png` - weights + activations aggregated per layer type.
+ `footprint_distribution.png` - histograms of the per-layer weights / activation / total footprint.
