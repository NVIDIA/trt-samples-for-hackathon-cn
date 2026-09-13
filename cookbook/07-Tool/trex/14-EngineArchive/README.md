# 14 - Engine Archive (TEA)

Bundle an engine and its analysis artifacts into a **TensorRT Engine Archive**
(`.tea`) - a single ZIP file. This is the cookbook re-implementation of
`trt-engine-explorer`'s `archiving.EngineArchive`.

A `.tea` bundles the engine plan (`engine.trt`) together with its graph / profile
JSON and a `plan_cfg.json` describing the engine (properties + IO tensors,
extracted by deserializing the plan), so a whole exploration session travels as
one file.

Deserializing the engine to extract `plan_cfg.json` requires **TensorRT** (the
engine must match this TensorRT version).

`EngineArchive(path, mode="w"|"r")`: `writef_txt` / `writef_bin` / `add_file` /
`archive_plan_info` for writing; `namelist` / `readf` for reading.

## Running

```bash
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # create and read back a .tea archive
```

## Output

+ `case_create_archive` - write `model.engine.tea` bundling the engine + JSON + plan info.
+ `case_read_archive` - reopen the archive and print the stored plan info (name, layers, IO tensors).
