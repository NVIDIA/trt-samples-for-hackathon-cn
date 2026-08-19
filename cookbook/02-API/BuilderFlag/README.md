# Builder Flag

+ Every `trt.BuilderFlag`, what it costs, and where in the cookbook each one is really demonstrated.

+ Steps to run.

```bash
python3 main.py
```

Measured on B200, **TensorRT 11.1.0.106** (20 flags).

## What this adds over `../BuilderConfig/`

[`../BuilderConfig/`](../BuilderConfig/README.md) shows the *shape* of the flag API — `set_flag`,
`get_flag`, `clear_flag`, the `flags` bitmask — using one flag as a stand-in placeholder. It never
builds with any of them, so it cannot say what any flag does.

This directory is about the flags themselves: all twenty, set one at a time against the same
network, with the consequence **measured**.

## 1. TF32 is already on, and assigning `flags` turns it off

```txt
flags on a freshly created BuilderConfig: 64 -> ['TF32']
after set_flag(REFIT)   : get_flag(REFIT) = True, flags = 80
after clear_flag(REFIT) : get_flag(REFIT) = False, flags = 64
after `flags = 1<<REFIT | 1<<DEBUG`: TF32 still on? False
```

`flags` is a plain bitmask, so assigning it **replaces** everything — including the default. That
one line silently drops TF32, which is a precision change nobody asked for, and the only symptom is
that your engine gets slower. Prefer `set_flag` / `clear_flag`.

(What TF32 being on actually costs you numerically is measured in
[`../../07-Tool/Onnxruntime/`](../../07-Tool/Onnxruntime/README.md): clearing it moves the
TensorRT-vs-ONNX-Runtime disagreement by **644x**.)

## 2. All twenty flags, one at a time

Every flag is *accepted* — not one fails the build. Only five change the plan at all:

| flag | plan size | delta |
| --- | --- | --- |
| *(baseline)* | 55,692 | — |
| `REFIT` | 82,284 | +26,592 |
| `REFIT_IDENTICAL` | 82,284 | +26,592 |
| `STRIP_PLAN` | 78,972 | +23,280 |
| `REFIT_INDIVIDUAL` | 55,964 | +272 |
| `VERSION_COMPATIBLE` | 104,900,812 | **+104,845,120** |
| *(the other 14)* | 55,692 | +0 |

**Acceptance is not effect.** The fourteen that show `+0` are not no-ops — they act at runtime, or
in the build log, or on hardware this machine does not have. A plan-size diff is simply the wrong
probe for them. That is exactly why the rest of this file, and `case_coverage_map`, exist.

## 3. `VERSION_COMPATIBLE` costs 100 MiB, and one flag takes it back

```txt
baseline                                             55,692 B
VERSION_COMPATIBLE                              104,900,812 B
VERSION_COMPATIBLE + EXCLUDE_LEAN_RUNTIME            55,692 B
EXCLUDE_LEAN_RUNTIME alone                           55,692 B
```

`VERSION_COMPATIBLE` embeds a **lean runtime** in the plan so another TensorRT version can load it.
On a 54 KiB engine that runtime is a **100 MiB** surcharge — it is a fixed cost, so the smaller the
model the more absurd the ratio.

`EXCLUDE_LEAN_RUNTIME` returns the plan to the baseline **exactly, byte for byte**: you keep version
compatibility and ship the runtime separately (see
[`../../04-Feature/LeanAndDispatchRuntime/`](../../04-Feature/LeanAndDispatchRuntime/README.md)).

And **alone it is a silent no-op** — no error, no warning, identical plan. It can only ever subtract
something `VERSION_COMPATIBLE` added.

## 4. `STRIP_PLAN` can make the engine bigger

The flag everyone reaches for to shrink an engine:

```txt
small network (32 conv filters)                      55,692 ->       78,972 B  (BIGGER)
weight-heavy network (+2.4 M matmul weights)      9,508,100 ->      109,532 B  (87x smaller)
```

On a small network the refit metadata it **adds** outweighs the weights it **removes**. It is worth
it exactly when weights dominate the plan — so check the ratio before adopting it, and remember the
weights have to come back via a refit at load time
([`../../04-Feature/WeightStripping/`](../../04-Feature/WeightStripping/README.md)).

## 5. What a "timing cache" is actually full of

`DISABLE_TIMING_CACHE` and `DISABLE_COMPILATION_CACHE` both leave the plan byte-identical, so the
matrix shows `+0` for each. Measuring build time is a bad probe here — on a network this small the
saving is a few percent and GPU contention swamps it. The **cache itself** is deterministic:

```txt
flags                                 build   timing cache after the build
(none)                                3.88s                     26,409 B
DISABLE_TIMING_CACHE                  3.16s                        198 B
DISABLE_COMPILATION_CACHE             3.14s                        376 B
```

**Both** flags empty it — which is the surprise, because only one of them has "timing" in its name.
Reading the three sizes as header + timings + compiled kernels:

| component | size |
| --- | --- |
| header alone (timing cache off) | ~198 B |
| + tactic timings | ~178 B |
| + JIT-compiled kernels | **~26,033 B (98.6%)** |

So the thing everyone calls "the timing cache" is overwhelmingly a **compilation** cache; the tactic
timings it is named after are a rounding error. That is also why `DISABLE_COMPILATION_CACHE` shrinks
a file with `timing` in its name — which looks like a bug report until you know the layout.

Turn either off to make a build reproducible
([`../../07-Tool/Polygraphy/More/12-TacticsAndReproducibility/`](../../07-Tool/Polygraphy/More/12-TacticsAndReproducibility/README.md))
or to prove a stale cache is the culprit. Build time is the cost, and on a real model it is minutes.

## 6. `MONITOR_MEMORY` writes to the log, not the plan

Plan size is identical with and without it. The flag is pure diagnostics: it makes the builder emit
memory-usage records into the logger, visible only at `INFO`/`VERBOSE` severity and only while
building. Reach for it when a build OOMs and you need to know which phase asked for the memory.

## 7. The coverage map

Most flags need a whole workflow to mean anything, and live elsewhere. Three are listed but not
demonstrated anywhere, and say so:

| flag | demonstrated in |
| --- | --- |
| `DEBUG` | listed only — build-time debug sync, no artifact to show; debug *tensors* are `04-Feature/DebugTensor` |
| `GPU_FALLBACK` | `04-Feature/DLAStandalone` — needs a DLA core, absent on this GPU |
| `REFIT` | `04-Feature/Refit` |
| `DISABLE_TIMING_CACHE` | here (`case_cache_flags`) + `04-Feature/TimingCache` |
| `EDITABLE_TIMING_CACHE` | `04-Feature/TimingCache` |
| `TF32` | `07-Tool/Onnxruntime` |
| `SPARSE_WEIGHTS` | `04-Feature/Sparsity` |
| `SAFETY_SCOPE` | `04-Feature/Safety`, `08-Advance/Safety` — QNX / DRIVE only |
| `DIRECT_IO` | `04-Feature/DataFormat` |
| `VERSION_COMPATIBLE` | here + `04-Feature/VersionCompatibility` |
| `EXCLUDE_LEAN_RUNTIME` | here + `04-Feature/LeanAndDispatchRuntime` |
| `ERROR_ON_TIMING_CACHE_MISS` | `07-Tool/Polygraphy/More/12-TacticsAndReproducibility` |
| `DISABLE_COMPILATION_CACHE` | here (`case_cache_flags`) |
| `STRIP_PLAN` | here + `04-Feature/WeightStripping` |
| `REFIT_IDENTICAL` | `04-Feature/WeightStripping` |
| `WEIGHT_STREAMING` | `04-Feature/WeightStreaming` |
| `REFIT_INDIVIDUAL` | `04-Feature/Refit` |
| `STRICT_NANS` | listed only — needs a NaN-producing graph; no dedicated example yet |
| `MONITOR_MEMORY` | here (`case_monitor_memory`) |
| `DISTRIBUTIVE_INDEPENDENCE` | listed only — needs a tensor-parallel group to mean anything |

**The assert is the point.** `case_coverage_map` checks this table against
`trt.BuilderFlag.__members__` and fails with the offending name if they diverge. A flag added by a
future TensorRT cannot quietly go undocumented, which is how "usage of each flag" stays true instead
of being true only on the day it was written.

## Related

+ [`../BuilderConfig/`](../BuilderConfig/README.md) — the rest of `IBuilderConfig`: memory pools,
  optimization level, tactic sources, profiling verbosity.
+ [`../Builder/`](../Builder/README.md) — `IBuilder` itself.
+ [`../../04-Feature/`](../../04-Feature/README.md) — the dedicated example for most of the flags
  above.
