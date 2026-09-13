# 11 - Process Engine Pipeline

The full trex workflow, end to end: **ONNX -> build -> profile -> JSON -> explore**.
This is the cookbook re-implementation of `trt-engine-explorer`'s
`utils/process_engine.py`, which drives `trtexec` to build and profile an engine
and generate all the JSON artifacts trex consumes.

Unlike the other trex examples (which share one prebuilt engine via `get_data.py`),
this example **builds its own engine**, so it **requires a GPU and `trtexec`**.

It ties together the pieces from the other examples:

| Step | Case            | What it does                                        | From example |
| ---- | --------------- | --------------------------------------------------- | ------------ |
| 1    | `case_build`    | build engine + export graph JSON (`trtexec`)        |              |
| 2    | `case_profile`  | profile engine + export profile / timing JSON       |              |
| 3    | `case_metadata` | parse trtexec logs into metadata JSON               | #09          |
| 4    | `case_draw`     | render the engine graph to SVG                      | #02          |
| 5    | `case_explore`  | load an `EnginePlan` and print summary + precisions | #00 / #09    |

## Running

```bash
python3 main.py   # runs the whole pipeline (needs a GPU + trtexec)
```

All artifacts are written to `pipeline_out/`.


> **Needs the Graphviz `dot` binary**, which is a **system** package rather than a pip one:
> `apt-get install graphviz`. The `graphviz` Python module only shells out to it, so without the
> binary every render raises `ExecutableNotFound` — the Python package alone is not enough, which is
> the easy mistake here. This example checks for it up front and prints
> `[SKIP] the Graphviz \`dot\` binary is not on PATH` instead of failing.
>
> Installed on this machine as of 2026-09-08: **graphviz 2.43.0** (`/usr/bin/dot`), so this case
> runs rather than skips.
