# FP16 Tuning

+ A tool to trade accuracy against performance layer by layer: build the model in FP16 with exactly
  one layer pulled back to FP32, run it, and record what that layer alone did to the error and to
  the latency. Repeating this for every tunable layer answers "which layer is responsible for my
  FP16 error" directly, instead of by bisection.

+ Steps to run.

```bash
python3 main.py
```

## It still works under TensorRT 11, but not the way it used to

Earlier versions of this tool set `BuilderFlag.PREFER_PRECISION_CONSTRAINTS` and pinned individual
TensorRT layers to FP32. **That flag was removed**, and a strongly-typed network takes its
precisions from the model rather than from the builder, so that approach is gone for good.

The tool was rewritten around the equivalent that strong typing does support: the precision
decision moves up to the **ONNX graph**. Each trial converts the model to FP16 with the chosen node
excluded (ModelOpt autocast, `convert_to_mixed_precision(..., nodes_to_exclude=...)`), then builds a
default strongly-typed engine from that graph. The engine's precision follows the ONNX dtypes, so
nothing has to be forced on the TensorRT side.

One consequence is visible in the parameters: `tune_type_list` holds **ONNX op types** -
`Conv`, `Gemm`, `MatMul` - and the rows of the report are ONNX node names, not TensorRT layer names.

## What comes out

Two reports, one per way of supplying input data, each rewritten after every layer so a long run can
be read while it is still going:

| File | Input data |
| ---- | ---------- |
| `FP16Tuning-data-report.md` | recorded, from `00-Data/data/InferenceData.npz` |
| `FP16Tuning-random-report.md` | random, generated at `infer_shape` |

A report looks like this - the first three rows are the baselines every other row is judged against:

```txt
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError | BestPerf | BestAcc |
|     1 | Pure FP32 🟩        | y            |          0.036 |             0 |          |         |
|     2 | Pure FP16 🟦        | y            |          0.028 |    3.5115e-05 |          |         |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.028 |    3.5115e-05 |          |         |
|     4 | node_conv2d         | y            |          0.028 |    3.5115e-05 | 1 🔴     | 2 🔴    |
|     7 | node_linear_1       | y            |          0.028 |    9.9754e-06 | 3 🔴     | 1 🔴    |
```

Row 1 is the accuracy reference and row 2 is the cost of going to FP16 everywhere; the rest say what
each layer is worth. `BestAcc` ranks by `MaxAbsError` on `focus_tensor` only - error on the other
outputs is reported but not ranked, because one tensor has to be chosen for the ordering to mean
anything. `BestPerf` ranks by measured `GPUTime`, which comes from a `trtexec` run of the built
engine and is therefore absent when `test_performance=False`. The ranking colour is just a band
(🔴 top 5, 🟠 top 10, 🟡 top 15, 🟢 top 20).

In the example above `node_linear_1` is the answer: holding that one layer in FP32 cuts the error
by 3.5x and costs nothing measurable.

Two log files also come out, and they are not the same thing:

+ `log-main.py.log` — stdout and stderr: case markers, the progress bar, TensorRT's own messages.
  This is what a unit-test sweep captures.
+ `FP16Tuning.log` — the tool's own narration, one line per step, named by the `log_file` parameter.
  Separate on purpose: tuning every layer of a real model produces far more text than anyone wants
  interleaved with the rest.

## Where the code is

The `FP16Tuning` class lives in `tensorrt_cookbook/utils_class.py`, where its parameters are
documented. `main.py` keeps only the two ways of driving it, so it reads as a worked example of the
arguments rather than as the implementation.

Pass `log_file` a file name — the class builds the logger from it; one logger is created per file,
so two sessions writing to the same file share it.

+ Thanks Xuewei Li for providing the original design of this tool.
