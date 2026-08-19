# QDQPlacementAutotune

+ ModelOptimizer's Q/DQ placement search, driven by **real TensorRT latency** rather than by a proxy.

+ The question it answers: given a model you have decided to quantize, *where* should the Q/DQ pairs
  sit? Move one across a fusion boundary and TensorRT builds a different set of kernels. There is no
  analytic answer — the search builds the engine and times it.

## Running

```bash
python3 main.py
```

Everything is generated locally, nothing is downloaded. Measured on **1 x B200**, TensorRT
11.1.0.106, `nvidia-modelopt` 0.44.0. Expect several minutes: every candidate scheme is a full
engine build plus a timing run.

## First, the blocker

The autotuner can time candidates two ways — through the TensorRT Python API (`TensorRTPyBenchmark`,
the **default**) or by shelling out to `trtexec` (`TrtExecBenchmark`, `--use_trtexec`). On TensorRT
11 the default one cannot start:

```
modelopt/onnx/quantization/autotune/benchmark.py:361
    self.network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
AttributeError: type object 'tensorrt.tensorrt.NetworkDefinitionCreationFlag'
                has no attribute 'EXPLICIT_BATCH'
```

`EXPLICIT_BATCH` was removed once explicit batch became the only mode; TensorRT 11.1.0.106 offers
only `STRONGLY_TYPED`, `PREFER_AOT_PYTHON_PLUGINS`, `PREFER_JIT_PYTHON_PLUGINS`. The autotuner
catches the `AttributeError` and reports just:

```
ERROR - Failed to initialize TensorRT benchmark
```

which does not name the cause. **Every run here therefore passes `--use_trtexec`.** This is the same
family of ModelOpt-vs-TRT-11 breakage as the `plugin_creator_list` shim in
[`05-Plugin/ONNXPTQWithPlugin`](../../05-Plugin/ONNXPTQWithPlugin/README.md) — see `99-Todo` §2.2.

## What the search buys

| | |
| --- | --- |
| baseline (INT8, default placement) | 0.110 ms |
| after autotuning | **0.090 ms** |
| speed-up | **1.174x** |
| cost | 138.1 s, 14 TensorRT benchmarks |

```
baseline.onnx        : 13 nodes, 0 Q / 0 DQ
optimized_final.onnx : 21 nodes, 4 Q / 4 DQ
```

Note what is being optimized. The tensors that get quantized do not change; what moves is **where
the Q/DQ pairs sit relative to the fusions**. The objective is a measured TensorRT latency, not a
proxy such as "quantize more operators". That is the difference from
[`07-Tool/FP16Tuning`](../FP16Tuning/README.md), which searches per-layer precision against
*accuracy*.

## Regions and pattern signatures

The unit of search is a **region** — a sub-graph treated as one placement problem:

| Size | Pattern | Schemes | Best |
| ---: | ------- | ------: | ---: |
| 3 | `Conv -> Relu -> Conv` | 4 | 0.0979 ms |
| 5 | `Conv -> Relu -> Conv -> Relu -> MaxPool` | 4 | 0.0938 ms |
| 13 | `COMPOSITE(...)` — the whole graph | 4 | 0.0938 ms |

`performance_threshold: 1.02` — a candidate must beat the incumbent by 2% to be adopted, so noise
alone does not move the answer.

A pattern **signature** is the op sequence *with its attributes*:

```
Conv[dilations=1x1,group=1,kernel_shape=3x3,pads=1x1x1x1,strides=1x1]->Relu->Conv[...]
```

so `Conv[kernel_shape=3x3]` and `Conv[kernel_shape=1x1]` are different patterns and never share a
cached result. That is what makes the cache safe to carry between models.

## The pattern cache — a measured negative

`autotuner_state_pattern_cache.yaml` from model A, reused when tuning a structurally similar model B
(same conv blocks, different head):

| Tuning model B | Benchmarks | Wall time | Speed-up found |
| -------------- | ---------: | --------: | -------------: |
| without the cache | 14 | 141.5 s | 1.420x |
| with model A's cache | 14 | 140.7 s | 1.388x |

**The cache saved nothing here** — 14 benchmarks either way. Both models are small enough that the
search exhausts its scheme budget regardless. The mechanism is real and correctly keyed; the benefit
needs a model with more repeated structure than this one.

**And do not read the speed-up column as a cache effect.** The search is stochastic — it mutates the
top-scoring schemes and stops at a scheme budget — so each run is a sample, not the answer. Across
two runs of exactly this pair the numbers were `1.420x / 1.388x` (above) and `1.421x / 1.149x`: the
cached row moved by **0.24** between runs while the uncached row barely moved at all. Only the
benchmark count is a fair comparison between those two rows.

## When this is worth running

When you are already committed to INT8/FP8, the model has repeated structure, and the deployment is
long-lived enough to amortise minutes-to-hours of search. It is a *build-time* cost that buys a
*run-time* win, and both are measured in the same currency — which is exactly why the number above
can be trusted, and why the two model-B rows cannot be compared to each other.
