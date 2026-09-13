# 12 - Excel Summary

Export an engine plan to an Excel workbook (`.xlsx`) with a summary sheet, the
per-layer table, precision statistics and embedded figures. This is the cookbook
re-implementation of `trt-engine-explorer`'s `excel_summary.ExcelSummary`,
rewritten with **openpyxl** (instead of pandas + xlsxwriter) and embedding
**Matplotlib** PNGs (instead of plotly images).

## Requirements

```bash
pip install openpyxl
```

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # write the Excel workbook (no GPU required)
```

## Output

`engine_summary.xlsx` with worksheets:

+ **Summary** - model / device / builder / performance key-value pairs.
+ **Layers** - the curated per-layer table (name, type, precision, tactic, latency, footprint).
+ **Precision** - byte totals per precision for input/output activations and weights.
+ **Latency chart** - an embedded latency-by-type figure.

`write_engine_excel(plan, path, image_files=...)` takes an optional
`{sheet_name: png_path}` dict of figures to embed.
