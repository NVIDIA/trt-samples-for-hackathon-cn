# trex - TensorRT Engine Explorer

Explore the structure and performance of a **built** TensorRT engine by analysing
the JSON files that TensorRT / `trtexec` can export. This is a cookbook port of
NVIDIA's [`trt-engine-explorer`](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer)
(a.k.a. `trex`), re-implemented **without pandas and plotly**:

+ the per-layer table is a plain `list` of `dict` (+ NumPy arrays via `plan.col(...)`),
+ figures are drawn with Matplotlib and **saved to files** (no notebook / browser / hover UI).

The shared utility code lives in `tensorrt_cookbook/utils_engine_explorer.py`.

## Input data

An engine plan is described by several JSON files, all GPU-free to analyse:

| File                    | Produced by                                      | Content                              |
| ----------------------- | ------------------------------------------------ | ------------------------------------ |
| `*.graph.json`          | `trtexec --exportLayerInfo` (detailed verbosity) | final inference graph: layers, tensors, tactics, precisions |
| `*.profile.json`        | `trtexec --exportProfile --dumpProfile`          | per-layer latency                    |
| `*.timing.json`         | `trtexec --exportTimes`                          | per-iteration end-to-end timing      |

`get_data.py` builds a small INT8-QAT MNIST engine and exports these files into the
shared `data/` directory. It is run automatically (via each example's `pre:` step)
and **skips rebuilding** when the files already exist, so all sub-examples share one
engine. Run it once by hand with `python3 get_data.py` if you want to inspect the JSON.

## Sub-examples

| Directory              | Feature                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `00-LoadEnginePlan`    | Load a plan (`EnginePlan`), print the engine summary and precision stats |
| `01-ReportCardLatency` | Latency report card: per-type / per-layer / distribution / timing plots  |

Each sub-example is independently runnable:

```bash
cd 00-LoadEnginePlan
python3 ../get_data.py   # build the engine + export JSON (needs a GPU; skipped if present)
python3 main.py          # analyse the JSON files (no GPU required)
```
