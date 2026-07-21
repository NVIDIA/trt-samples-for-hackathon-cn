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
