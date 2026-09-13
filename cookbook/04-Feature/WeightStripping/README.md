# WeightStripping

+ Build a weight-stripped engine with `BuilderFlag.STRIP_PLAN` and refit it back to a full one from the original ONNX.

+ The idea: if the ONNX file is shipped anyway, the weights inside the engine are a duplicate. `STRIP_PLAN` keeps the optimized kernels and schedule but drops the weights, and the engine is refitted at load time from that same ONNX.

+ Steps to run.

```bash
python3 main.py
```

## What the example measures

| Case | What it shows |
| ---- | ------------- |
| `case_full_engine` | Baseline: a normal engine and its output |
| `case_stripped_engine` | Size of the stripped plan, the output **before** refitting, and the output after refitting from ONNX |
| `case_strip_and_refit_identical` | `STRIP_PLAN` combined with `REFIT_IDENTICAL` |

Typical output on this machine (MNIST CNN):

```txt
Full engine:      12.6MiB
Stripped engine:  201.0KiB
Saved  12.4MiB, i.e. 98.4% of the full plan
Output before refitting: [0. 0. 0. 0. 0.]
[check] before refit:False,maxAbsDiff=1.815e+01,...
Output after refitting:  [-5.088133  -2.3532653  6.905129   7.531381  -9.480327 ]
[check] after refit:True,...
```

## Three things worth noticing

+ **A stripped engine is not a usable engine.** It deserializes and executes without complaining, it
  just computes with absent weights and returns all zeros. The `[check] before refit:False` line in
  the log is *expected* — it is the demonstration, not a failure. Never ship a stripped plan without
  the ONNX it was built from.

+ **`STRIP_PLAN` implies refittable.** `BuilderFlag.REFIT` does not need to be set as well.

+ **Refitting restores the accuracy, not necessarily bit-identical results.** `STRIP_PLAN` prevents
  the builder from specializing on weight values it is about to discard, so the stripped plan may
  select different kernels from the full one. The two then differ by ordinary floating-point noise —
  observed here as anything between exactly `0` and a few `1e-3` on logits of magnitude ~10,
  varying from build to build because tactic selection is timing-based. That is why the check uses a
  tolerance rather than exact equality.

## `REFIT_IDENTICAL`

`REFIT_IDENTICAL` promises the builder that the weights refitted later will be *the same values* it
saw at build time. That lets it keep weight-dependent optimizations, so the refitted engine performs
like a normal one. It is meant for shipping one set of weights to several inference backends or GPU
architectures. Refitting such an engine with **different** weights is undefined behaviour.

## Related examples

+ `04-Feature/Refit` — the general refit APIs, including refitting from raw weight arrays and the
  other `OnnxParserRefitter` entry points. It sets `STRIP_PLAN` among several other flags at once;
  this example isolates it.
+ `04-Feature/WeightStreaming` — a different problem. There the weights are too large for device
  memory and get streamed in during inference; here they are removed from the plan entirely and
  restored before inference.
