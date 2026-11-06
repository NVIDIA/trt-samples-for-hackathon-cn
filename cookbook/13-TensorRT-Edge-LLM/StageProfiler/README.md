# StageProfiler

+ An `IProfiler` that groups per-layer times by a caller-declared **stage**.

```bash
python3 main.py
```

Adapted from `cpp/profiling/layerProfiler.{h,cpp}` of
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) (Apache-2.0), where the stages are
an LLM's prefill and decode passes. Nothing is copied: theirs is a C++ singleton holding
`LayerProfile{name, stageName, timeMs[], iterationCount}`; this is the same data model in Python.

## Why a stage dimension

`CookbookProfiler` reports one flat stream of `(layer, time)` pairs, which is right when an engine
is run one way. It stops being right the moment the same engine is driven in two regimes — two
optimization profiles, two input shapes, a cold call and a warm one — because the two sets of times
land in one list and the resulting averages describe a workload that never ran.

`case_what_a_flat_profiler_loses` measures exactly that:

```txt
    One flat bucket :    93.0 us
    Separated       : first-call 151.8 us, steady 91.5 us
    The flat mean sits between the two and describes neither (91.5 < 93.0 < 151.8).
```

## What the stages are here

MNIST on a B200 has no prefill and decode, so the example uses the contrast this engine really has.

**Cold versus warm is large and reproducible.** `case_cold_and_steady`:

```txt
  stage 'first-call': 11 layers, 1 iteration(s), 154.8 us total
  stage 'steady':     11 layers, 20 iteration(s), 85.8 us total
    first-call 154.8 us vs steady 85.8 us -> 1.80x
```

Measured over three separate builds while writing this: **1.81x, 1.65x, 1.64x**.

**Batch 1 versus batch 4 is a negative result, and it is kept.** `case_two_input_shapes`:

```txt
    batch1 87.1 us vs batch4 88.6 us -> 1.02x
```

Four times the work for the same time: this engine is launch-bound, not compute-bound, and an
MNIST network at batch 4 does not come close to filling a B200. The point is not the non-result —
it is that **establishing two regimes are indistinguishable is itself an answer**, and a flat
profiler cannot give it either way.

## Two design decisions

**Un-staged times are kept, not dropped.** Anything recorded outside a `with profiler.stage(...)`
lands in `(unlabelled)`. The warm-up calls in `case_two_input_shapes` show up there with 124.6 us,
which is a feature: a warm-up silently folded into the measured numbers is how the flat profiler
goes wrong in the first place.

**Not a singleton.** Upstream deletes the copy constructor and assignment operator and exposes
`LayerProfiler::getInstance()`, because a C++ runtime has several execution contexts that must
report into one table. Here an ordinary object is enough and easier to reason about — TensorRT
calls `report_layer_time` on whatever object `context.profiler` points at.

## Status

Deliberately standalone for now, rather than folded into `tensorrt_cookbook.CookbookProfiler` or
[`04-Feature/Profiler/`](../../04-Feature/Profiler/README.md). Whether to merge it is an open
question — see [`../../99-Todo/trtellm.md`](../../99-Todo/trtellm.md).

Measured on TensorRT 11.3.0.99, B200, 2026-10-12.
