# Weight Streaming

+ Run a model whose weights do not fit in VRAM, by keeping most of them in host memory and choosing a device budget.

+ Steps to run.

```bash
python3 main.py
```

An engine normally holds all its weights in device memory. Weight streaming lifts that: the weights
live in host memory and are fetched as each layer needs them, so the device footprint becomes a
**budget you choose** rather than a property of the model. The cost is bandwidth.

Two APIs, not interchangeable:

| API | When | What it is |
| --- | --- | --- |
| `enable_weight_streaming=True` | compile | **permission** — the engine is built so streaming is possible |
| `torch_tensorrt.runtime.weight_streaming(module)` | run | the context manager exposing `device_budget` |

Without the compile-time flag the runtime knob does nothing.

Measured on B200, TensorRT 11.1.0.106, `torch_tensorrt` 2.14.0a0, 12 × `Linear(4096, 4096)` in FP16
= **384 MiB of weights**, batch 8, median of 10:

| budget | resident | latency | vs no streaming | max abs diff vs Torch |
| --- | ---: | ---: | ---: | ---: |
| (disabled) | 384.0 MiB | 0.159 ms | 1.00x | – |
| 100% | 384.0 MiB | 0.156 ms | **0.98x** | 1.19e-07 |
| 50% | 192.0 MiB | 4.329 ms | 27.20x | 1.19e-07 |
| 25% | 96.0 MiB | 6.154 ms | 38.67x | 1.19e-07 |
| 10% | 38.4 MiB | 6.754 ms | 42.44x | 1.19e-07 |
| 0% | 0.0 MiB | 7.479 ms | **47.00x** | 1.19e-07 |

## What the numbers say

**A full budget is free.** At 100% the streaming-enabled engine is as fast as the ordinary one
(0.98x, i.e. within noise). Compiling with `enable_weight_streaming=True` therefore costs nothing
until you actually lower the budget — it is worth enabling on any model that might one day need it.

**The cliff is immediate and steep.** Halving the budget costs **27x**, not 2x. This is not a smooth
trade: as soon as the working set does not fit, every inference re-fetches weights over PCIe, and
the GEMMs go from compute-bound to transfer-bound. Weight streaming is not a tuning knob for
performance; it is the difference between *running* and *not running*.

**The numerics never change.** `max |diff|` against eager PyTorch is 1.19e-07 at every budget
including zero. Streaming moves bytes, it does not approximate — so a model validated at one budget
stays valid at another.

## Why not Llama-2

The upstream example (`.../weight_streaming_example.py`) drives this with a gated Llama-2 download.
The cookbook does not download at run time, and what matters for streaming is the **ratio of weights
to activations** — a stack of large GEMMs has exactly the shape of a transformer's feed-forward path,
which is where the weight mass of an LLM sits. A 384 MiB model is enough to show the whole curve.

## Related

+ [`../../../04-Feature/WeightStreaming/`](../../../04-Feature/WeightStreaming/README.md) — the same
  feature at the TensorRT API level, without the Torch frontend.
+ [`../../../04-Feature/WeightStripping/`](../../../04-Feature/WeightStripping/README.md) — the other
  way to shrink an engine: ship it without weights and refit them at load.
+ [`../EngineCaching/`](../EngineCaching/README.md) — the compile-time cost this pairs with.
