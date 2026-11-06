---
name: trt-debug-mismatch
description: >-
  Find where a TensorRT engine's output starts to disagree with its reference,
  by bisecting the pipeline to the first divergent boundary. Use when a TRT
  result differs from ONNX Runtime, PyTorch or a saved golden output. Triggers:
  TensorRT output is wrong, results differ from ONNX Runtime, TRT vs PyTorch
  mismatch, accuracy drop after conversion, engine gives different numbers,
  max abs diff too large, cosine similarity dropped, output is NaN, which layer
  is wrong, FP16 overflow, golden output mismatch, TRT regression after upgrade.
  NOT for build failures (the engine does not build), NOT for performance
  (`trt-perf-analysis`), NOT for migrating weak to strong typing
  (`trt-strong-typing-migration`).
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  version: "1.0"
  tags:
    - debugging
    - accuracy
    - numerics
    - bisection
---

# Find the First Divergent Boundary

A TensorRT engine produces different numbers from its reference. The goal is not
to explain the final difference — it is to find **the first boundary at which the
two pipelines stop agreeing**, because everything after that point is a
consequence, not a cause.

Method adapted from the `debug-trt-mismatch` skill in
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
(Apache-2.0), reframed for the cookbook's tools.

## The rule that matters most

**Never weaken the check to make the mismatch disappear.** Raising a tolerance,
shrinking the input set, switching to a laxer metric, or changing the seed are
all ways to stop seeing the problem without fixing it. If you believe the check
itself is wrong, demonstrate that separately and say so explicitly — do not
quietly relax it as part of the investigation.

## Step 1 — Pin the reproducer before changing anything

Record, in the report you will eventually write:

- TensorRT version (`python3 -c "import tensorrt; print(tensorrt.__version__)"`),
  CUDA version, driver, GPU name and compute capability;
- the exact model file and its hash, the exact input (file or seed), the build
  flags, and the optimization profile and shape actually used at run time;
- the reference: which runtime, which version, which precision.

A mismatch that only reproduces sometimes is a different problem — see
*Non-determinism* below — and must be identified as such before bisecting.

## Step 2 — Separate the four stages

Most "TensorRT is wrong" reports are not about TensorRT. Check these in order and
stop at the first one that fails:

| Stage | Question | Cookbook tool |
| --- | --- | --- |
| Input | Do both pipelines receive byte-identical input? | `check_array(a, b)` on the input arrays |
| Export | Does the ONNX file already disagree with the framework? | `07-Tool/Onnxruntime/`, `07-Tool/CheckTorchOperator/` |
| Build | Did the engine build in the precision you think? | `04-Feature/EngineInspector/`, `trtexec --exportLayerInfo` |
| Run | Right profile, right shapes, right bindings? | `07-Tool/ContextPrinter/` |

Ordering is not optional. A preprocessing difference at stage 1 produces a
plausible-looking layer-level divergence at stage 4 and sends you to the wrong
place entirely.

## Step 3 — Quantify the disagreement properly

Use `check_array` rather than a single number:

```python
from tensorrt_cookbook import check_array
check_array(output_trt, output_reference, b_agreement=True, b_cosine=True, b_relative=True)
```

Read these together, not individually:

- **`nonfinite` first.** A NaN or Inf count above zero makes every other statistic
  meaningless. Find where the first one is produced before reading anything else.
- **`maxAbsDiff` large but `cosine` ≈ 1** — a scale or a single outlier, not a
  structural error.
- **`cosine` well below 1** — the shapes of the two results differ; suspect a
  layout, transpose, or wrong-weights problem.
- **`agreement` low with small `maxAbsDiff`** — ordinary accumulated rounding,
  often benign for FP16.
- **All statistics tiny but the task output is wrong** — look at argmax/topk
  behaviour, not at the tensor.

Compare against the *same* reference at the *same* precision. An FP16 engine
compared with an FP32 reference is measuring two things at once.

## Step 4 — Bisect to a layer

Once you know the divergence is inside the engine:

1. **Mark intermediate tensors as outputs** and compare them. When you do not own
   the builder, `11-TensorRT-Model-Connect/GraphTransform/` adds an output without
   editing it.
2. **Use debug tensors** (`04-Feature/DebugTensor/`) to dump values at run time
   without changing the graph's outputs.
3. **Walk forward, not backward.** Find the earliest tensor that disagrees. The
   last one always disagrees and tells you nothing.
4. **Bind by the name the engine reports**, not the name the network used — these
   differ after fusion, and `GraphTransform/README.md` shows a measured example.

Expect fusion to limit the resolution: a Myelin-fused region has no intermediate
tensors to inspect, and the boundary you find is the region, not a layer inside
it.

## Step 5 — Attribute the cause

Common causes, cheapest test first:

| Suspected cause | Test |
| --- | --- |
| FP16 range overflow | Rebuild that region in FP32; if it fixes it, find the overflowing tensor rather than keeping the FP32 pin blindly |
| Q/DQ placement | `07-Tool/QDQPlacementAutotune/`, `02-API/Layer/QDQStructure/` |
| Wrong weights / layout | Compare the weight tensor itself against the checkpoint, not the activation |
| Plugin | Run the plugin's kernel standalone on the same input |
| Dynamic shape | Does it reproduce at the profile's `opt` shape? Different shapes select different kernels |
| Non-determinism | Run the same engine twice on the same input. If those two disagree, the reference is not the problem |

## Step 6 — Report

State, in this order:

1. the exact reproducer (Step 1);
2. the first divergent boundary, with the measured numbers at that boundary;
3. the evidence that ruled out each earlier stage;
4. the smallest fix, and where it belongs;
5. **what you did not check** — other shapes, other precisions, other hardware.

That last item is not a formality. A fix verified at one shape on one GPU is a
fix verified at one shape on one GPU, and saying so is part of the result.
