# 09 - Parse trtexec Log

Parse trtexec **build** and **profiling** log files into the metadata JSON files
that carry the device properties, builder configuration and performance summary.
This is the cookbook re-implementation of `trt-engine-explorer`'s
`utils/parse_trtexec_log.py`.

trtexec logs are organized into `=== Section ===` blocks of `[ts] [I] key: value`
lines. `parse_build_log` / `parse_profiling_log` extract:

+ build log -> `model_options`, `build_options`, `device_information`
+ profiling log -> `performance_summary`, `inference_options`, `device_information`

Feeding the resulting metadata back into `EnginePlan` (via
`profiling_metadata_file` / `build_metadata_file`) fills the Device / Builder /
Performance sections of `print_summary`, which are empty when only the graph +
profile JSON are supplied.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON and logs (needs a GPU; skipped if present)
python3 main.py          # parse the logs (no GPU required)
```

## Output

+ `case_parse_logs` - print key device / build / performance fields extracted from the logs.
+ `case_write_metadata` - write `model.build.metadata.json` and `model.profile.metadata.json`.
+ `case_summary_with_metadata` - load the plan WITH metadata; `print_summary` now shows the
  device properties, builder configuration and performance summary.
