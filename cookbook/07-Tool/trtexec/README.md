# trtexec

+ Command-line tool of TensorRT, attached with an end-to-end performance test tool.

+ Installation: attached with TensorRT, the executable program is /opt/bin/trtexec

+ Steps to run (introduction is included in the script).

```bash
chmod +x main.sh
./main.sh
```

All numbers below were re-measured on this machine 2026-09-06 (**B200**, **TensorRT 11.1.0.106**) with
`00-Data/model/model-trained.onnx`, the MNIST CNN.

## Steps in main.sh

| Step | Topic                                                                 | Output        |
| :--: | --------------------------------------------------------------------- | ------------- |
|  01  | Run from an ONNX file with no extra option                            | result-01.log |
|  02  | Parse ONNX, build and save an engine with the common build options    | result-02.log |
|  03  | Load the engine and run inference                                     | result-03.log |
|  04  | Print engine information (`--dumpLayerInfo` / `--exportLayerInfo`)    | result-04.log |
|  05  | Print profiling information (`--dumpProfile` / `--exportProfile` / `--exportTimes`) | result-05.log |
|  06  | Save input / output data (`--dumpOutput` / `--dumpRawBindingsToFile`) | result-06.log |
|  07  | Run with loaded input data (`--loadInputs`)                           | result-07.log |
|  08  | Build and run with a plugin (`--plugins`)                             | result-08.log |
|  09  | Build a second engine at another optimization level, export its profile | result-09/10.log |
|  10  | Post-process the exported JSON with `parse_export_json.py`            | result-11.log |
|  11  | Accuracy checking against reference outputs                           | result-12.log |
|  12  | Global performance tuner / build-route search                         | result-13/14.log |
|  13  | Strongly typed network, and the precision flags that no longer exist  | result-15.log |
|  14  | Throughput with several inference streams                             | result-16.log |
|  15  | Weight-stripped engine                                                | result-17.log |

## parse_export_json.py

`main.sh` has always *produced* `--exportProfile` / `--exportTimes` JSON without ever reading it
back. This script closes that loop:

+ `--exportProfile` → per-layer time table ranked by total time, plus a total row.
+ Two `--exportProfile` files → per-layer A/B comparison with a `% difference` column, so the cost
  of a build option (here `--builderOptimizationLevel`) is visible layer by layer. Layers that
  exist in only one of the two runs are reported as `new` / `gone` rather than silently dropped —
  which is exactly what the shipped example shows, since level 1 and level 5 pick different
  fusions and therefore emit different layer names.
+ `--exportTimes` → the first N per-iteration H2D / compute / D2H rows, plus min / mean / P50 / P90
  / P95 / P99 / max latency recomputed offline.

```bash
python3 parse_export_json.py --profile=model-trained-exportProfile.json \
                             --times=model-trained-exportTimes.json \
                             --reference=model-trained-reference-exportProfile.json \
                             --threshold=5
```

This is the cookbook re-implementation of TensorRT-OSS `samples/trtexec/{profiler,tracer}.py`.
Re-implementing also fixes a real incompatibility: the upstream tracer reads the pre-TensorRT-10
key names `startInMs` / `inMs` / `outMs`, while current trtexec writes `startH2dMs` / `h2dMs` /
`d2hMs`. Both spellings are accepted here.

## Accuracy checking

trtexec compares its outputs against golden data itself, so a simple numerical check no longer
needs Polygraphy.

| Option                  | Meaning                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| `--accuracyAlgorithm`   | `l0` (fraction outside `atol`/`rtol`), `l1` (MAE), `l2` (MSE), `lInf` (max abs error), `cos` (1 - cosine similarity) |
| `--atol` / `--rtol`     | Absolute / relative tolerance, only used by `l0` (both default `1e-5`)      |
| `--accuracyThreshold`   | Fail the run if the computed accuracy loss exceeds this value               |
| `--loadRefOutputs`      | Reference outputs, in the same `name:file` form as `--loadInputs`           |
| `--refPair=N`           | Group one `--loadInputs` with one `--loadRefOutputs`                        |

Two rules that are not obvious from the help text, both of them measured:

+ **`--accuracyThreshold` is mandatory**, not optional, as soon as `--loadRefOutputs` or
  `--refPair` is used: `[E] --accuracyThreshold (with a positive value) is required when
  --loadRefOutputs or --refPair is set.`, followed by the entire help text.
+ **`--refPair` needs at least two pairs.** A single `--refPair=0` fails with
  `[E] When using --refPair, you need at least two pairs of I/O.` For one pair, drop the option
  and pass `--loadInputs` / `--loadRefOutputs` on their own.

The five algorithms are **not comparable to each other**. Feeding input `a` while asking for the
reference output of input `b`, the same mismatch reads:

| Algorithm | Accuracy loss for tensor `y` |
| --------- | ---------------------------- |
| `l0`      | 1.000000                     |
| `l1`      | 0.001432                     |
| `l2`      | 0.000003                     |
| `lInf`    | 0.003492                     |
| `cos`     | 0.001497                     |

So a threshold tuned for one algorithm means nothing for another: at `--accuracyThreshold=1e-3`
this mismatch fails under `l0`, `l1`, `lInf` and `cos`, and **passes** under `l2`.
Step 11C runs the failing case on purpose — a check that has never been seen to fail is not
evidence that the check works.

## Global performance tuner

Instead of hand-sweeping build options, trtexec searches a **build-route expression** itself.

| Option                  | Meaning                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| `--tuneBuildRoutes=<expr>` | The search space. `-knob=[a\|b\|c]` is a variable knob, `-knob=fixed` pins one, knobs are space-separated |
| `--tuneBuildRouteFile=<f>` | Read the same expression from a newline-delimited file                   |
| `--tuningSearch=<spec>` | `fast` = baseline + one-off variation per knob (linear, default); `full` = full Cartesian product; `mixed` = fast scan then exhaustive over the knobs that helped |
| `--tuningCacheFile=<f>` | JSON-lines file that per-iteration results are appended to                  |
| `--tuningTimeOut=<s>`   | Stop after N seconds; `-1` = no timeout                                     |
| `--continue`            | Resume an interrupted loop from `--tuningCacheFile`                         |
| `--saveAllEngines`      | Keep every iteration's engine as `<engine>.iter<N>` (needs `--saveEngine`)   |
| `--dryRun`              | List the enumerated routes and exit without building                        |
| `--helpBuildRoute[=knob]` | Dump the knob database as JSON (optionally filtered) and exit             |
| `--setBuildRoute=<route>` | Single-shot build of one specific route — this is what the tuning loop forks child workers with, so any result reproduces standalone |

**The knobs are internal compiler knobs, not trtexec build options.** `--helpBuildRoute` returns
209 of them here (`tuner_version` 2.19.45), things like `-conv_lowering=[on|off]`,
`-kgen:tiling=[0|1|2]`, `-cask_fusion:num_tactics=int`. Passing an ordinary trtexec option is
rejected outright:

```txt
[E] Failed to parse --tuneBuildRoutes expression: Unknown knob: -builderOptimizationLevel
```

`--tuningSearch` decides how the expression is expanded — with the two knobs used in `main.sh`
(2 values × 3 values):

| `--tuningSearch` | Routes enumerated |
| ---------------- | ----------------- |
| `fast`           | 4 (baseline, then one knob varied at a time) |
| `full`           | 6 (Cartesian product) |

The tuning cache is JSON-lines: the first line is metadata — including `default_build_route`, the
value of **all 209 knobs**, which is the easiest way to see what TensorRT's defaults actually are —
and then one line per iteration with `build_route` and `gpu_time`.

**Honest result on this model:** the 4 routes land within about **1 %** of each other, and the
winner is not even stable between runs — two runs of this very script reported
`-conv_lowering=on -kgen:tiling=2` (28.36 us, spread 1.00 %) and
`-conv_lowering=off -kgen:tiling=2` (28.38 us, spread 1.39 %). For a small MNIST CNN the search
costs about a minute and buys nothing: the run-to-run noise is larger than the effect being
searched for. The tuner earns its wall-clock on models where fusion decisions actually differ.

## Strongly typed, and the precision flags that are gone

Upstream `samples/trtexec/README.md` Example 6 presents `--stronglyTyped` as the way to opt in to
`kSTRONGLY_TYPED`, and warns not to combine it with `--int8` or `--best`. On TensorRT 11 both
halves of that advice are obsolete:

+ `--stronglyTyped` is a **no-op**: the default build already prints `Precision: Strongly Typed`
  (see `result-02.log`, which passes no such flag).
+ `--fp16`, `--int8`, `--best`, `--precisionConstraints` and `--layerPrecisions` **do not exist**
  any more — `[E] Unknown option: --fp16`.

That is a cleaner failure than the Polygraphy CLI, which still advertises `--fp16` / `--int8` in
its help and only throws from inside `CreateConfig` when a build is attempted; see
[`../Polygraphy/README.md`](../Polygraphy/README.md) and
[`../Polygraphy/More/13-PerLayerPrecision/`](../Polygraphy/More/13-PerLayerPrecision/README.md).

## Throughput with several inference streams

Upstream Example 5. `--infStreams=N` runs N execution contexts concurrently (`--streams=N` is
still accepted as the older spelling the upstream README uses):

| `--infStreams` | Throughput   | Median GPU compute time |
| -------------: | ------------ | ----------------------- |
| 1              | 15464.3 qps  | 0.0590 ms               |
| 2              | 26451.2 qps  | 0.0632 ms               |
| 4              | 26274.9 qps  | 0.0762 ms               |

**This conclusion changed on B200.** On H100 the same table read 15588.9 / 26358.6 / **42919.2** qps
— 2.75x throughput at 4 streams for 1.42x latency. On B200 the second stream still pays for itself
(**1.71x** throughput for 1.07x latency) but **the fourth buys nothing at all**: 26274.9 qps is
*below* the 2-stream figure while latency keeps climbing to 1.29x. The engine is a small MNIST CNN;
two streams already saturate what this GPU will overlap for it, and the rest is pure queuing.
**A stream count tuned on one GPU does not transfer.**

trtexec itself warns that once the streams overlap the latency numbers stop being meaningful,
so `Throughput` is the only metric to read here.

## Weight-stripped engine

`--stripWeights` keeps the kernels and the schedule but drops the weights
(`--stripAllWeights` is the alias for `--refit --stripWeights`):

```txt
13250084 B  model-trained.trt
  157308 B  model-trained-stripped.trt      -> 84x smaller
```

Two things worth knowing, both measured in step 15:

+ **A stripped engine loads, runs, and reports `PASSED` while producing all zeros.** Nothing warns
  that the weights are missing.
+ **Refitting it back does not work from the CLI on this build.** `--refitFromOnnx` emits no log
  line and leaves the output at zero, and `--dumpRefit` prints nothing at all. The working path is
  the Python one: [`../../04-Feature/WeightStripping/`](../../04-Feature/WeightStripping/README.md)
  refits the same model through `trt.Refitter` + `trt.OnnxParserRefitter` and gets the original
  output back exactly. The ONNX-side counterpart is
  [`../Polygraphy/Surgeon/`](../Polygraphy/Surgeon/README.md) (`surgeon weight-strip`).

## Notes on the upstream sample

`samples/trtexec/` in TensorRT-OSS is the reference for this directory. Three places where it no
longer matches the shipped tool, all of them checked against the local checkout:

| Upstream | Status on TensorRT 11.1 |
| -------- | ----------------------- |
| `tracer.py` reads `startInMs` / `inMs` / `outMs` | trtexec writes `startH2dMs` / `h2dMs` / `d2hMs`; `parse_export_json.py` accepts both |
| README Example 5 uses `--streams` | still accepted, but the documented spelling is now `--infStreams` |
| README Example 6 recommends `--stronglyTyped`, warns against `--int8` / `--best` | `--stronglyTyped` is a no-op, the precision flags no longer exist |

DLA (upstream Example 2) is not covered here: this machine has no DLA.
