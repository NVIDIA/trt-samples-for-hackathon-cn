# Refit

+ Swap new weights into a compiled module without recompiling — and the one way it goes silently wrong.

+ Steps to run.

```bash
python3 main.py
```

Compilation is the expensive step. When only the *weights* change — a new checkpoint, a different
LoRA adapter, a fine-tuned head — rebuilding wastes all of it, because the graph is identical and
only the numbers inside it moved. `torch_tensorrt.dynamo.refit_module_weights(compiled, exported)`
replaces them in place.

Measured on B200, TensorRT 11.1.0.106, `torch_tensorrt` 2.14.0a0, 6 × `Linear(512, 512)`:

| operation | seconds | vs recompile |
| --- | ---: | ---: |
| first compile (refittable) | 6.48 | – |
| full recompile, new weights | 5.39 | 1.0x |
| **`refit_module_weights`** | **0.37** | **14.7x** |

The refitted engine matches the new eager model, and differs from the old one, which is what makes
the speed-up meaningful rather than just fast.

## `immutable_weights=False` is required, and enforced

The default builds an immutable engine. Refitting one is properly refused, with a message that says
what to do:

```txt
AssertionError: Refitting is not enabled. Please recompile the engine with immutable_weights=False.
```

## The dangerous part: a different graph is **accepted**

Refit replaces weights, not structure — so a model with a different architecture should not be a
valid source. It is not rejected. Refitting a **6-layer** engine from an **8-layer** model:

```txt
ACCEPTED, no error
    vs the true 8-layer output      : max |diff| = 7.811e-02   <- WRONG
    vs that model's first 6 layers  : max |diff| = 0.000e+00   <- exactly this
```

**The extra two layers are silently discarded.** The engine keeps its own structure and takes as
many weights as it has slots for. There is no exception and no warning, and the output is plausible
— it is a real model's output, just not the model you asked for.

The third comparison is what turns "the numbers differ" into an explanation: matching the truncated
model *exactly* (0.0) identifies the mechanism, where a bare mismatch would only have said something
was wrong.

**Graph identity is the caller's precondition to enforce.** A service that refits from checkpoints
it did not build must check the architecture itself — comparing `state_dict()` keys and shapes before
refitting is enough, and is the check this API does not do for you.

## Related

+ [`../../../04-Feature/Refit/`](../../../04-Feature/Refit/README.md) — the same feature at the
  TensorRT API level, where weights are named and set individually.
+ [`../../../04-Feature/WeightStripping/`](../../../04-Feature/WeightStripping/README.md) — shipping
  an engine without weights and supplying them at load time.
+ [`../WeightStreaming/`](../WeightStreaming/README.md) — the other way to decouple weights from the
  engine, by keeping them in host memory.
