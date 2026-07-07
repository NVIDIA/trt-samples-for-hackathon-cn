# 07 - Compare Engines

Compare two (or more) TensorRT engine plans - here an **INT8** engine vs an
**FP16** engine built from the same MNIST network. This is the cookbook
re-implementation of `trt-engine-explorer`'s `compare_engines.py`
(`compare_engines_overview` / `compare_engines_summaries_tbl`).

`get_data.py` builds both engines (`model.*` = INT8, `model.fp16.*` = FP16). The
original renders interactive plotly dropdowns; here each view is a Matplotlib
figure saved to a PNG file, plus a printed comparison table.

To compare more than two engines, add entries to the `engines` list at the top
of `main.py`.

## Running

```bash
python3 ../get_data.py   # build both engines + export JSON (needs a GPU; skipped if present)
python3 main.py          # compare the engines (no GPU required)
```

## Output

+ `case_summary_table` - side-by-side summary (layers, average time, weights, activations) and overall speedup.
+ `compare_latency_by_type.png` - stacked latency-by-type bar per engine.
+ `compare_latency_by_type_grouped.png` - grouped bar comparing per-type latency, plus a per-type speedup table.
+ `compare_latency_by_precision.png` - stacked latency-by-precision bar per engine.
