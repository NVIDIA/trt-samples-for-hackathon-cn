# 99-Todo

+ Todo list and research notes for the cookbook.

## 1. Backlog — what is left

One item. **M4 and M11 landed on 2026-09-07**; only P5 is left, and it is blocked on a
Torch-TensorRT upgrade rather than on us.

+ **M4** -> [`07-Tool/QDQPlacementAutotune/`](../07-Tool/QDQPlacementAutotune/README.md) — Q/DQ
  placement search timed by real TensorRT latency: **1.174x** (0.110 -> 0.090 ms) for 14 engine
  builds. Two honest negatives recorded with it: the pattern cache **saved nothing** on models this
  small (14 benchmarks either way), and the search is **stochastic** — the cached model-B row read
  1.388x on one run and 1.149x on another, so a cache-vs-no-cache latency difference cannot be read
  as an effect of the cache.
+ **M11** -> folded into
  [`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT`](../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
  as `case_autocast_bf16` + `case_node_sensitivity` — BF16 is **6.0x further from FP32** than FP16
  on a model that never overflows (opset 19 -> 22), and the `data_max` sweep shows three thresholds
  producing the **byte-identical graph** with different reported errors.

Completed items are not listed here — each one's findings live in its own example README, and the
lessons worth carrying across examples are in §2. Everything done so far is in the working tree on
`wili/dev` and **not yet committed**.

| #   | Item | Source | Target | Ready? |
| --- | ---- | ------ | ------ | ------ |
| P5  | `Input(profiles=[...])` — N optimization profiles on one engine, runtime selection by index or `"auto"` | `.../multi_optimization_profiles.py` | `06-DLFrameworkTRT/Torch-TensorRT/OptimizationProfiles/` | **BLOCKED, re-verified again 2026-09-07**: `torch_tensorrt` 2.14.0a0 still has none of it — `Input` has no `profiles` parameter, `torch_tensorrt.runtime.optimization_profile` does not exist. `set_active_profile` lives in `core/runtime/TRTEngine.cpp`, so it needs a full source build, not a Python overlay. The TensorRT-API equivalent already exists as `08-Advance/MultiOptimizationProfile` |

Also still open, from §3.1: **S8** — re-measure the remaining H100-labelled numbers on B200. Read
the thermal lesson in §2.1 before starting; it changes how that has to be done.

### 1.0 Engine visualization: converge on two systems

Three implementations currently turn an engine layer-info JSON into a picture, and the plan is to
end with two. The comparison table lives in
[`07-Tool/EngineVisualization/README.md`](../07-Tool/EngineVisualization/README.md); this is the work.

**V1. Control flow — DONE.** `export_engine_as_onnx` no longer refuses a `Loop` engine, and
represents **all 8** dependencies of `model-loop` (6 forward + 2 back-edges) where the other two
writers keep 5 and say nothing. Two modes, both covered by `07-Tool/EngineVisualization/
export_as_onnx.py` on `model-loop.onnx` and `model-for.onnx`:

+ **default** leaves the back-edge as a real cycle. `onnx.checker` rejects that — ONNX is defined as
  single static assignment — but **Netron draws it correctly**, and these files are read rather than
  executed. Verified with a hand-built three-node cyclic ONNX, rebuilt by the test rather than
  committed.
+ **`b_break_cycle=True`** renames each write to an SSA version and adds a `BackEdge` marker node,
  giving a DAG that passes `onnx.checker`. Needed by anything built on `onnx_graphsurgeon`, which
  raises `Cycle detected in graph!` otherwise. Not the default because it reads worse: the markers
  are dead ends you have to resolve by name, and removing the back-edge often splits the picture
  into disconnected components.

Engines without control flow are byte-identical in both modes.

Still open is the *structured* form: an ONNX `Loop` with a real subgraph body, which would let
Netron fold the body away. This is harder than it first looked — **the engine has no subgraph to
recover.** TensorRT compiles the loop into a flat instruction stream with an explicit `cjmp` layer
and a back-edge, so producing an ONNX `Loop` means decompiling: inferring the body boundaries from
the branch and mapping recurrences onto carried dependencies. Nested loops, multiple exits and an
`If` inside the body all complicate it, and a subtly wrong reconstruction is worse than the honest
cycle we now emit, because it looks correct. Not scheduled.

**V1b. The other two writers silently drop edges, and this is not a control-flow bug.** Both
`EngineVisualization/main.py` and `utils_engine_explorer.py` build their producer index with
`producer[name] = index` (a later write overwrites the earlier) and then skip self-loops with
`source != index`. Any engine reusing a tensor name loses edges with no warning; a `Loop` engine
loses 3 of 8. At minimum they should detect a name with several producers and say so. Deliberately
left alone for now so the ONNX path could be finished first.

**V2. Absorb `trex/02-DrawEngineGraph`'s colouring into `EngineVisualization`, then delete it. DONE.**
The writers now do everything `trex/02` did, and a few things it did not: nodes filled by layer
type, edges coloured by tensor precision, per-layer latency with the slowest layer outlined in red,
binding nodes, the display toggles, and rasterisation to `.svg` alongside the `.gv`. Beyond `trex`
they also keep every JSON field in a node tooltip, dash the back-edges of a loop, and filter
multi-profile engines with a printed count instead of silently.

The palette had to be **rebuilt rather than copied**: `layer_colormap` and the old `TYPE_COLOUR`
both use the capitalised TRT 8/9 names, while TensorRT 11 emits `kgen`, `gemm`, `correlation`,
`maxpool`, `memset`, `add`, `cjmp`, `custom_layer`. Measured here, the old palettes coloured at most
two of the five types in an MNIST engine, so both tools were drawing a nearly grey graph.

`fold_no_ops` is the one feature deliberately **not** taken. A `NoOp` in a built engine usually
marks a reformat or a copy the builder could not eliminate, so removing it makes the picture
prettier by deleting the evidence — and it conflicts with the rule the ONNX writer follows, which is
not to silently change the structure of what you were asked to draw.

`07-Tool/trex/02-DrawEngineGraph/` was removed on 2026-09-10 and its references swept. The helper
it called, `render_engine_graph` in `utils_engine_explorer.py`, **stays**: `trex/11-ProcessEngine
Pipeline` draws a graph as one step of its pipeline, so only the example that existed solely to draw
graphs is gone. Discovery went 256 -> 255 cases.

Note this removes only `trex/02-DrawEngineGraph`. The rest of `07-Tool/trex/` — the report cards —
is unaffected and stays.

### 1.1 Next: the TensorRT 11.2 upgrade

The 11.0 work is complete: **257 / 257** on 2026-09-08 (`tests/logs/REPORT-2026-09-08.md`). The
checklist below is what the upgrade has to deal with, in priority order.

**1. The version labels are already wrong, and the upgrade makes that worse.** 46 example READMEs
say **TensorRT 11.1.0.106**; the container currently has **11.0.0.114**, and will have 11.2. Those
numbers *were* genuinely measured on 11.1.0.106, so they were never relabelled — relabelling without
re-measuring is the one thing this repo refuses to do. After the upgrade they are two versions stale.
Decide deliberately between:
  + re-measure and relabel (correct, expensive — this is S8 all over again), or
  + add a single dated "measured on" line per README and leave the numbers (cheap, honest).

Do **not** let a bulk find-and-replace happen.

**2. `04-Feature/RefitObserver` should be able to drop its external TensorRT.** It currently builds
against an unpacked **11.3** tarball via `TRT_OBSERVER_PATH` because `IRefitterObserver` landed in
11.2 and the system install predates it. On 11.2 the system headers should suffice — retire the
`TRT_OBSERVER_PATH` indirection and the `LD_LIBRARY_PATH` requirement in its `unit_test.yaml`.
Still C++ only: the 11.3 cp312 wheel had no `set_refit_observer`, so re-check the 11.2 wheel before
assuming a Python binding exists.

**3. Re-check the API-removal fixes.** Two examples broke on 11.0's weak-typing removal and were
migrated during the sweep. 11.2 may remove more:
  + `02-API/Layer/QDQStructure/main.py` — `BuilderFlag.INT8` (removed) and `add_quantize` /
    `add_dequantize` now requiring an explicit `output_type`;
  + `07-Tool/trtexec/main.sh` — `--fp16` / `--int8` / `--best` / `--precisionConstraints` all
    rejected as "Unknown option".
  The `trt-strong-typing-migration` skill is the reference. Its `scripts/migrate.py --write`
  **reformats the whole file and strips the SPDX header** (it round-trips through AST unparse), so
  use it for its diff and verdict, then edit by hand.

**4. Reinstall the pip extras first**, every time — see §1.2.

**5. P5** is still blocked on Torch-TensorRT, not on TensorRT; the upgrade will not move it.

### 1.2 Provisioning: the container resets between sessions

`tensorrt_rtx`, `tensorrt-lean` and `tensorrt-dispatch` do **not** survive a reset and must be
reinstalled before any sweep:

```bash
pip install tensorrt_rtx tensorrt-lean tensorrt-dispatch
```

Graphviz (`/usr/bin/dot`, apt) has survived so far. **A green sweep does not prove the extras ran**:
the 16 `optional-package` cases guard their imports and exit 0 with a `[SKIP]` line, so an
un-provisioned run still reports 257/257. Verify afterwards by grepping each log for the `^[SKIP]`
marker — not for the word "skip", which appears in `10-TensorRT-RTX/01-ComputeCapability`'s normal
output ("skip setting") and gives a false positive.

## 2. Findings to carry forward

### 2.1 Measurement lessons

Three mistakes recurred often enough to be worth stating once. Each produced a **confident and
wrong** result that looked exactly like a real measurement.

**Thermal throttling.** GPU 0/1 on this machine reach 88-91 C under sustained load and drop from
~1900 MHz to **352 MHz** (`clocks_throttle_reasons.active = 0x20`, SW thermal slowdown, at 437 W
against a 1000 W limit). The same engine measured 2.8 ms on a cool GPU and 11.0 ms on a hot one; one
sweep reported **111.8 ms where the answer was 10.9 ms**, and moved an apparent crossover by a factor
of two. `08-Advance/ContextParallelism` now picks the coolest GPUs, waits for them, and prints the SM
clock next to every latency. **Any latency number taken back-to-back on this box is suspect unless
the clock is recorded with it.**

**Orchestration cost swamping the measurement.** `08-Advance/MultiTask` and
`06-DLFrameworkTRT/Torch-TensorRT/DistributedInference` both first measured concurrency as a
*regression* (0.35x, 0.65x, 0.79x) because two Python barriers per iteration cost more than the GPU
work being timed. The fixes were to scale the work up and to measure throughput (barrier paid once)
rather than per-round latency. `MultiTask` now measures the barrier floor explicitly and prints it
beside every result. **When per-iteration work approaches the cost of the synchronisation around it,
the benchmark is no longer about TensorRT.**

**Comparisons rigged by the choice of probe.** `04-Feature/OnnxPTQMethod` compared entropy-vs-max
calibration only on the outlier-bearing data the outliers had been *planted* in -- where `max` wins
by construction. Adding an ordinary probe **reversed the ranking**. Separately,
`04-Feature/LowBitQuantization` asserted "block scaling beats bit count" against data showing the
opposite. **Where a comparison can be rigged by the choice of input, measure both sides of it.**

A fourth, smaller one: `08-Advance/ResourceProbe` asserted the specific teardown symptom it happened
to observe. That symptom is non-deterministic (exit 0 with 5 logged errors, or a silent SIGSEGV), so
the assertion was flaky. Assert the invariant, not the observation.

### 2.2 Upstream defects and workarounds we are carrying

These are **not fixed** — the examples route around them. Each should be re-tested after the next
ModelOpt / Torch-TensorRT / TRT upgrade, and the first one should be reported upstream.

| What | Where it bites | What the example does instead |
| ---- | -------------- | ----------------------------- |
| **ModelOpt's Q/DQ autotuner cannot use its own default benchmark backend on TRT 11.** `autotune/benchmark.py:361` reads `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`, removed once explicit batch became the only mode. The `AttributeError` is caught and reported only as `Failed to initialize TensorRT benchmark`, naming nothing | `07-Tool/QDQPlacementAutotune` | pass `--use_trtexec`, which takes the `TrtExecBenchmark` path and works. **Report to ModelOpt** alongside `plugin_creator_list` |
| **ModelOpt 0.44.0 is incompatible with TRT 11.** TRT 11 removed `IPluginRegistry.plugin_creator_list`, which ModelOpt still reads | `05-Plugin/ONNXPTQWithPlugin` | carries `apply_modelopt_trt11_shim()`, which re-adds the property as an alias of `all_creators`. **Report to ModelOpt** — the shim exists to let the example run, it is not a fix |
| **`trt_plugins=` is unusable here.** Even past the shim, ONNX Runtime rejects a graph containing a TensorRT custom op **at session load**; `calibration_eps=["trt"]` does not help. Three errors, none of which names its own cause | `05-Plugin/ONNXPTQWithPlugin` | calibrate an ONNX-expressible stand-in, then transplant the resulting Q/DQ onto the plugin graph |
| **Sub-8-bit never reaches an engine.** Torch-TRT's converter rejects NVFP4 / MXFP8 / INT4-AWQ outright; the ONNX route fails for **all five** formats; exporting a ModelOpt-quantized torch module to ONNX fails for every format; INT4 weight-only quantizes but fails the TRT build on block-size divisibility (`inputSize % scaleSize == 0`, 3211264 / 25600) | `04-Feature/LowBitQuantization`, `04-Feature/OnnxPTQMethod` | ship the negative result. **Only INT8 and FP8 reach an engine on this stack** |
| **TRT-LLM plugins refuse CUDA 13**, so tensor parallelism cannot be demonstrated | `06-DLFrameworkTRT/Torch-TensorRT/DistributedInference` | data parallel only (3.11x on 4 GPUs); TP is documented as blocked |
| **`GroupNormalizationPlugin` was removed** (TRT ≥ 10.7, Blackwell) | `07-Tool/OnnxGraphSurgeon/14_fold_exporter_subgraphs.py` | emit the native ONNX `GroupNormalization` (opset 21) instead of the plugin node |
| **E8M0 block scales cannot build.** Isolated with a 2x2 test — a bare Q/DQ pair builds fine; the E8M0 *scale* is the blocker, and FP4 block QDQ is rejected outright | `02-API/Layer/QDQStructure/block_quantization.py` | documented as the measured boundary |

### 2.3 Silent failures — the most valuable output of this round

None of these raise. Every one of them will produce a plausible number and a wrong answer.

+ **`refit_module_weights` accepts a structurally different model.** Refitting a 6-layer engine from
  an 8-layer module is **accepted**; the engine then matches that module's **first 6 layers exactly**
  (diff 0.0) and its true output not at all. The extra layers are discarded without a word. This is
  the most dangerous item on the page — `06-DLFrameworkTRT/Torch-TensorRT/Refit`.
+ **`use_explicit_typing=True` together with `enabled_precisions=...` is accepted silently**, not
  rejected. I predicted an error and was wrong — `06-DLFrameworkTRT/ModelOptimizer`.
+ **Binding a device pointer to a shape tensor is a SIGSEGV, not an error.** Shape tensors are read
  on the host, and nothing checks the pointer — `02-API/ShapeTensor` (the case runs in a subprocess
  precisely because it kills the process).
+ **`MAX` / `MIN` reductions silently drop NaN** where NumPy propagates it (8.0 / 5.0 vs `nan`), so a
  max-reduction is **not** a NaN detector — `04-Feature/CornerCase`.
+ **TensorRT can log 5 destructor errors and still `exit 0`** when its CUDA context is destroyed
  first — invisible to any exit-code-based check. And the symptom is non-deterministic: sometimes
  exit 0 with the errors, sometimes a silent SIGSEGV — `08-Advance/ResourceProbe`.
+ **Identical weights are deduplicated in the plan**, collapsing an 18 MiB engine to 2 MiB and making
  any load/streaming measurement built on repeated weights meaningless — `08-Advance/MultiDevice`.
+ **`polygraphy run --trt --fp16` under `POLYGRAPHY_USE_TENSORRT_RTX=1` drops the flag and passes.**
  The identical request through `CreateConfig(fp16=True)` raises. Exit-code-only checks cannot tell
  a dropped flag from an honoured one -- `07-Tool/Polygraphy/TensorRTRTX`.
+ **TF32 is on by default and is the first suspect when a hand-built network "does not match".**
  It put 2.643e-04 between a correct network and torch where float32 gives 8.941e-08, a **2956x**
  difference that reads exactly like a transposed blob -- `02-API/Network`, `04-Feature/RefitObserver`.
+ **`trt.Weights` wraps a pointer.** Values are read at `build()`, not at `add_*()`. Mutating the
  array in between silently changes the engine; letting it be collected is a use-after-free with no
  message -- `02-API/Network/weight_transport.py`.
+ **TensorRT-RTX's `set_compute_capability` returns `False` instead of raising** when
  `num_compute_capabilities` has not been assigned, and the build then quietly targets the current
  device -- `10-TensorRT-RTX/08-DeployTimeRuntimeConfig`.
+ **`IRuntimeConfig` does not keep its runtime cache alive.** Dropping the last Python reference
  after `set_runtime_cache()` segfaults the process, silently -- same example.
+ **Two error patterns can share the same max-abs-diff** while one has 80/80 elements wrong and the
  other 1/80. A single scalar tolerance cannot tell them apart —
  `07-Tool/Polygraphy/DebugWorkflow`, `07-Tool/AccuracyCheck`.

## 3. Open candidates

### 3.1 Standing items (not from any repo survey)

| #   | Item                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Note                                                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S8  | **在 B200 上重测所有标着 H100 的实测数字。9/10 完成，2026-09-06。** 重测并改写了 10 处中的 9 处。**最后一处 `07-Tool/NsightSystems` 被环境阻塞**：`nsys` 目前在这台机器上对**任何**目标都挂死 —— `nsys profile -o /tmp/x /bin/true` 不返回，GPU 全程 0%、日志为空、不产出 report；加 `--sample=none --cpuctxsw=none` 仍挂，`perf_event_paranoid=-1`，所以不是 CPU 采样、不是 TensorRT、也不是并发争用，是 nsys 自身。该例的数字**保持 H100 标签不动**（改标签而不重测等于造假），README 里已写明原因，等 nsys 恢复后再补。**四条结论因平台变化而改变**：(1) `08-Advance/GreenContext` —— **148 SM** 而非 114（`minSmPartitionSize`/`smCoscheduledAlignment` **仍是 8**），噪声邻居下加绿色上下文后 p95 = **0.99x**，即与独占无异（H100 上是 2.41x）；build-in-partition 收益 19% → **8%**（三次复现，0.010 ms vs 0.000–0.001 ms 噪声）。(2) `07-Tool/trtexec` —— `--infStreams=4` 在 H100 上是 2.75x 吞吐，B200 上**完全无收益**（26274.9 qps 反而低于 2 流的 26451.2），流数调优不跨 GPU 迁移。(3) `07-Tool/OnnxVisualization/90-Research/01-SubgraphInONNX` —— flat 与 loop **不再逐比特相同**（MaxDiff 0 → 1.19e-07~1.76e-05）；flat 的 build 爆炸从 2.0x 变成 **8.6x**（9.3 s → 79.4 s），折叠成 `Loop` 更划算。(4) `06-DLFrameworkTRT/Torch-TensorRT/EngineCaching` —— 原文「冷缓存下 build+save 会略慢」**在旧的 H100 数据上就不成立**（三行里两行更快），已改写为「中间那列被 warm-up 主导，不要当成开销读」。其余重测但结论不变：`08-Advance/MIG`（profile 表换成 `7g.180gb`/148 SM 至 `1g.23gb`/18 SM）、`08-Advance/TensorRTGraphSurgeon`（2.13x → 1.84x）、`05-Plugin/TritonAOTPlugin`、`07-Tool/TritonServerDeploy`（13.14 → 7.23 ms）、`07-Tool/OnnxVisualization`（3 处共用表） | 全部在 1965 MHz / 26 °C 下测，见 §2.1 |

### 3.2 TensorRT OSS — `/work/trt/TensorRT-GitHub`

**All intake from this repository is closed (2026-09-04).** G1, G3, G6, G7, G8, G10, G11 and G12
landed; G2, G4, G5 and G9 were declined and moved to §4.

**G12 leftovers, deliberately not taken.** G12 was a grab-bag; two parts were done and the rest are
recorded here rather than left dangling:

+ **Done** — `topkLastDimPlugin` as a perf footnote (measured: 2.12x at k=64, but **0.83x at k=8**,
  and `ITopKLayer` refuses k > 3840 by returning `None` from `add_topk`); and the `bfloat16.*`
  comparison, which found a real defect — `numpy_fp32_to_bf16` **truncated** instead of
  round-to-nearest-even and so disagreed with `torch.to(torch.bfloat16)` on **49.8%** of a random
  array. Fixed and vectorised (13x faster, now bit-exact against torch, NaN preserved).
+ **Not taken** — `streamReader.h` (the cookbook already has `FileStreamReader` in
  `cookbookHelper.cuh`), `ErrorRecorder.h` (already `CookbookErrorRecorder`), `safeCudaAllocator.h`
  (safety runtime, and `04-Feature/Safety` already covers the boundary),
  `plugin_utils.py::CudaCtxManager` and `common_runtime.py::ArrayWithOwner` (both are ownership
  idioms for sample plumbing the cookbook wrappers already own), and the cross-compilation
  Dockerfiles (`ubuntu-26.04`, `ubuntu-cross-aarch64`) — **unverifiable on this machine**, since
  nothing here can run an aarch64 target.

**Deliberately skipped as near-duplicates** — `quickly_deployable_plugins` (≈ `QuickDeployablePlugin`),
`onnx_custom_plugin` (≈ `ONNXParserWithPlugin`), `non_zero_plugin` (≈ `DataDependentShape`),
`deploy_to_triton` (≈ `07-Tool/TritonServerDeploy`), `sampleIOFormats` (≈ `04-Feature/DataFormat`),
`sampleNamedDimensions` (≈ `LabeledDimension`), `sampleProgressMonitor`, `sampleDistCollective`,
`network_api_pytorch_mnist`, `samples/python/refactored/`, `stream_writer` (≈ `02-API/Builder` +
`CookbookStreamWriter`), `common.py::setup_timing_cache` (≈ `04-Feature/TimingCache`).
Most C++ `plugin/*` dirs (bertQKV, efficientNMS, groupNorm …) are **deprecated since 10.12–10.15**;
11.0 additionally *removed* `batchTile`, `clip`, `coordConvAC`, `cropAndResize`, `gelu`, `leakyRelu`,
`normalize`, `singleStepLSTM`, `specialSlice`, `split`, `nms`, `proposal`. `tools/pytorch-quantization`
and `tools/tensorflow-quantization` are legacy (superseded by ModelOpt).

### 3.3 TensorRT internal (GitLab) — **vet before use**

The GitLab repo **no longer tracks `samples/`** (`oss_components.yml` lists `samples/**/*` as
`detracked`), and `plugin/` + `tools/trtexecCommon/` are identical to GitHub's. The GitLab-exclusive
surface is `tests/unitTests/`, `tools/`, `samples_internal/`, `testing/`, `scripts/`,
`plugin_internal/`, `projects/`, `documentation/`.

> Internal unit tests `#include` internal harness headers. The *layer APIs* are public, but any
> intake must be **rewritten from scratch against the public API — copy the idea, never the file.**

**Every L item is decided.** L1, L2, L3, L5, L6, L7, L8, L9 and L11 landed; L4 and L10 were
declined and moved to §4. Nothing from this repository is left undecided.

### 3.4 Model-Optimizer — `/work/trt/repos/Model-Optimizer`

Baseline: `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` already covers **AutoCast FP16**,
**torch INT8 QAT**, and **FP8 ONNX PTQ (max calibration)** on a tiny MNIST CNN. Do not re-propose
those. Since TRT 11 strong typing removed weak-typing INT8 calibration, ModelOpt is the sanctioned
quantization path, and every candidate below emits explicit Q/DQ a strongly-typed engine consumes.

**Every M item is decided (2026-09-06).** M4 and M11 are in §1; M5, M6, M8, M9 and M10 were
declined and moved to §4. Nothing from this repository is left undecided.

Skipped: `vllm_serve`, `llm_eval`, pure HF/vLLM serving (never touch TensorRT); `deepseek`,
`gpt-oss`, `minimax_m3`, `megatron_bridge`, `speculative_decoding`, `alpamayo`, `puzzletron`,
Minitron pruning (massive downloads / NeMo-Megatron containers).

### 3.5 TensorRT-RTX and tensorrtx

`10-TensorRT-RTX/{01..07}` already cover the RTX-only APIs **individually**; `02-API/Layer/Scale`
covers `add_scale` mechanics; `02-API/Network` only pokes at Network-object attributes.

**Every R and T item is decided (2026-09-06).** R1, R2 and T2 landed; R3, T1, T3 and T4 were
declined and moved to §4. Nothing from these two repositories is left undecided.

### 3.6 Torch-TensorRT and Tripy leftovers

Eight Torch-TensorRT examples landed under `06-DLFrameworkTRT/Torch-TensorRT/`; Tripy landed as
`07-Tool/nvtriPy`.

**Every P item is decided.** P1, P2, P3 and P4 landed, P5 is still blocked (§1); P6 was declined and moved to §4.
The blocker on P5 is unchanged and is recorded with the item:

**P5 is blocked on the installed build.** `torch_tensorrt` 2.14.0a0 has none of the API:
`Input(profiles=...)` raises `ValueError`, `torch_tensorrt.runtime.optimization_profile` does not
exist, `torch.classes.tensorrt.Engine` exposes no profile methods, and
`dynamo/conversion/_TRTInterpreter.py` hard-codes a single `create_optimization_profile()`. The
source clone carries the feature under the *same* `2.14.0a0` version string, but `set_active_profile`
lives in `core/runtime/TRTEngine.cpp` — overlaying the newer Python files would not work, it needs a
full source build. Revisit after a Torch-TensorRT upgrade. The TensorRT-API-level equivalent already
exists as `08-Advance/MultiOptimizationProfile`.

### 3.7 Community repos — ideas, not imports

| Repo                                                                                                                                                                                                                      | Idea worth stealing                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) (4.9k★)                                                                                                                                                           | the **per-op converter registry** as a Network-API teaching device                  |
| [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO), [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)                                                                          | NMS/post-processing plugins + fused CUDA pre/post as a realistic detection pipeline |
| [mmdeploy](https://github.com/open-mmlab/mmdeploy) (3.1k★)                                                                                                                                                                | multi-backend deploy + custom TRT plugins                                           |
| [jetson-inference](https://github.com/dusty-nv/jetson-inference) (8.9k★)                                                                                                                                                  | embedded C++ runtime demos                                                          |
| [WhisperLive](https://github.com/collabora/WhisperLive) (4.1k★), [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion), [SD-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) | streaming-ASR / diffusion end-to-end framing, dynamic-shape engine management       |

Not TRT-specific, ecosystem context only: TNN, Tengine, lite.ai.toolkit, YOLOX, yolov5.

---

## 4. Decided against — do not re-propose

+ **DLA.** Every DLA path (`tutDLA*.cpp`, `samples/sampleCudla`, a `04-Feature/DLA/` group) would
  take the "no DLA core available" early return on this B200 (`builder.num_DLA_cores == 0`) and
  could never be verified. **Shipping unverifiable examples is worse than shipping none.** Reopen
  only on Orin / DRIVE / Jetson; the API to build from was confirmed present in TRT 11.0:
  `trt.MemoryPoolType.DLA_{MANAGED_SRAM,LOCAL_DRAM,GLOBAL_DRAM}`;
  `IBuilderConfig.{default_device_type, set_device_type, get_device_type, is_device_type_set,
  reset_device_type, can_run_on_DLA, DLA_core}` with `trt.DeviceType.DLA`;
  `Runtime.{DLA_core, num_DLA_cores}`; `trt.EngineCapability.DLA_STANDALONE`.
+ **demoDiffusion.** A faithful port means large downloads, a multi-engine pipeline manager and
  FP8/FP4 calibration data. Point users upstream instead of shipping a trimmed copy that drifts.
+ **Parsing the `TRT_*` custom operators from ONNX.** `TRT_Attention` / `TRT_MoE` /
  `TRT_KVCacheUpdate` are already covered as network-API layers in `02-API/Layer/*`.
+ **`sampleDevice.h`-style C++ RAII wrappers.** Tried, added to `cookbookHelper.cuh`, refactored
  `08-Advance/CudaGraph` onto it, then reverted in full. The C++ examples keep explicit
  `cudaMalloc` / `cudaStreamCreate` / `cudaGraph*` calls paired with explicit releases.
+ **TensorRT-LLM intake — closed 2026-08-28.** Only 9 non-3rdparty files in that repo still
  `import tensorrt`; the engine-build stack (`builder.py`, `network.py`, `module.py`,
  `python_plugin.py`, `tools/plugin_gen/`) is deleted and all three plugin examples fail to import.
  For a Python-written TRT plugin take OSS `samples/python/python_plugin` (G3) instead.
+ **Two "gaps" that were not gaps:** `05-Plugin/InPlacePlugin` **does** already use aliased I/O in
  C++ (`v_2_0::IPluginV3OneBuild` + `getAliasedInput`; an earlier grep missed the camelCase), and
  `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` **does** already cover AutoCast.
+ RTX Flux.1[dev] demo and the tensorrtx model zoo: cite, never import.
+ **Five more candidates, closed 2026-09-04** (were L4 / L10 / M7 / M12 / P6):
  + **L4 — "Picky FP8"** (`tests/unitTests/tutPickyFP8.cpp` → 02-API/Layer/QDQStructure). An FP8
    Q/DQ/MatMul network deliberately engineered to be numerically sensitive to wrong types or
    scales (alpha = 1+2⁻¹¹ tricks). It is a *test* for TensorRT rather than a lesson about it: the
    payload is the adversarial constant, and reproducing it teaches nothing a reader can reuse.
    FP8 Q/DQ placement itself is covered by `02-API/Layer/QDQStructure` and now by
    `07-Tool/OnnxFP8QDQConvert`.
  + **L10 — Operator reference docs as a QA source** (`documentation/operators/*.rst`). Explicitly
    "not content to import": the idea was to cross-check `02-API/Layer/*/README.md` against the
    authoritative per-operator text. It is a documentation-review chore with no artefact, and the
    same guarantee is already enforced mechanically by `check_api_coverage` in the layer examples,
    which fails when TensorRT's API and the example disagree.
  + **M7 — Diffusion (SDXL / FLUX) INT8/FP8/FP4 PTQ** (`examples/diffusers/quantization/`).
    Multi-GB weights and ≥48 GB memory, and the TensorRT-specific content — strong typing plus
    multi-input dynamic profiles — is already carried by
    `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` and `08-Advance/MultiOptimizationProfile`
    without the download. See also the `demoDiffusion` entry above, which was declined for the same
    reason.
  + **M12 — Diffusion sparse attention + quantization-aware distillation**
    (`examples/diffusers/{sparsity,distillation}/`). Cutting-edge but heavy, and it is a *model
    research* technique rather than a TensorRT one: nothing in it exercises an API the cookbook
    does not already show.
  + **P6 — ResNet50 / NanoGPT / Stable Diffusion / SAM2 / ModelOpt quantization in Tripy**
    (`TensorRT-Incubator/tripy/{notebooks,examples}/`). Gated or multi-GB downloads *plus*
    `torch`+`transformers` inside the nvtripy venv, for a pre-1.0 API. `07-Tool/nvtriPy` already
    covers Tripy itself.
+ **Four TensorRT-OSS candidates, closed 2026-09-04** (were G2 / G4 / G5 / G9 in §3.2):
  + **DeBERTa end-to-end** (`demo/DeBERTa` → 03-Workflow). PyTorch→ONNX, ONNX-GS surgery to insert a
    disentangled-attention plugin, TRT vs ORT. Cost is dominated by "first get the model", and the
    plugin-insertion-by-graph-surgery idea is already carried by G6 (PackNet) on a model that needs
    no gated download.
  + **Refit ONNX via GS node replacement, BiDAF** (`samples/python/engine_refit_onnx_bidaf` →
    04-Feature/Refit). Replace unsupported nodes (HardMax/Compress), build refittable, refit
    fake→real weights. `04-Feature/Refit` already covers the refit API itself; what this adds is a
    specific model's unsupported-op workaround.
  + **Detectron2 Mask R-CNN R50-FPN** (`samples/python/detectron2` → 03-Workflow). Convert / run /
    validate, already strongly typed. Same "first get the model" cost as DeBERTa, with a heavier
    dependency (detectron2 itself) and nothing TensorRT-specific that a smaller model cannot show.
  + **Declarative data download** (`samples/python/downloader.py` → 00-Data). `download.yml`
    manifest + MD5 `verifyChecksum` + retries. This is infrastructure, not a TensorRT lesson, and it
    now points the wrong way: since 2026-09-01 the examples deliberately **do not download at run
    time** at all (see `00-Data/`), so a better downloader has nothing left to do.
+ **M5 — FastNAS structured pruning, CIFAR ResNet** (`examples/pruning/cifar_resnet.ipynb`),
  closed 2026-09-06. The upstream notebook stops at torch, so the whole TensorRT half — export,
  build, and a dense-vs-pruned comparison — would have to be written from scratch, on top of an
  actual FastNAS search plus fine-tuning run. Two of the lessons it would have carried are already
  on the page without the training cost: the "structured pruning changes tensor shapes, 2:4 sparsity
  only changes their contents" contrast is recorded as the L8 negative result (the builder *declines* sparsity here), and "a proxy metric is not latency" is exactly what M4 measures with
  real TensorRT timings. Pruning stays absent from the cookbook as a deliberate choice.
+ **Eight more candidates, closed 2026-09-06** (were R3 / T1 / T3 / T4 / M6 / M8 / M9 / M10):
  + **R3 — hand-built vs parsed under the mandatory RTX `STRONGLY_TYPED` flag**
    (`TensorRT-RTX/samples/helloWorld/`). The two-ways-to-build-one-net comparison is already
    `01-SimpleDemo`, and RTX's strong-typing requirement is already stated in `10-TensorRT-RTX`.
    R1 carries everything RTX-specific that is worth showing.
  + **T1 — LeNet-5 built through the network API from raw weights** (`tensorrtx/lenet/lenet.py`).
    Layer-by-layer network construction is already covered exhaustively by `02-API/Layer/*`, and the
    `addFullyConnected` -> MatMul+bias migration is a TRT-8-era note. What was actually novel here is
    the weight-transport convention, which is being taken separately as **T2**; once T2 lands, T1 is
    a second model built out of APIs the cookbook already documents.
  + **T3 — BatchNorm folded into `IScaleLayer`** (`tensorrtx/resnet/resnet34.cpp`). The content is an
    algebraic identity, not a TensorRT lesson: `02-API/Layer/Scale` already documents the API, and
    every supported path into TensorRT (ONNX parser, ModelOpt, Torch-TensorRT) folds BatchNorm
    automatically, so the manual fold is only needed by hand-built networks.
  + **T4 — YOLO anchor-grid decode inside a plugin** (`tensorrtx/yolov5/plugin/yololayer.cu`). Its own
    entry already said "do not port the code": it is `IPluginV2IOExt`, deprecated in TRT 10. Rewriting
    it on `IPluginV3` would be a new detection plugin, not an import, and `05-Plugin` already teaches
    `IPluginV3` on examples that can be verified without a YOLO checkpoint.
  + **M6 — ResNet-50 INT8 QAT with `mto.save`/`restore`** (`examples/cnn_qat/torchvision_qat.py`).
    Overlaps the MNIST QAT already in `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT`; the only
    genuinely new part is quantizer-state save/restore, which does not justify swapping in an
    ImageNet-style dataset tree that the no-runtime-download policy would then have to carry.
  + **M8 — ONNX PTQ for non-classifiers, SAM2 / Whisper** (`examples/windows/onnx_ptq/`). ORT- and
    DirectML-centric, so most of the work is TensorRT glue the upstream sample does not have, on top
    of multi-GB weights. The PTQ lesson itself is already `04-Feature/OnnxPTQMethod`.
  + **M9 — 2:4 structured sparsity via SparseGPT** (`examples/llm_sparsity/weight_sparsity/`).
    Llama2-7B, gated, ~44 GB of GPU memory. It would also land on top of a **measured negative
    result**: the builder *declines* sparsity on this machine (L8), so the example could not
    show a benefit even with the sparse weights in hand.
  + **M10 — LLM PTQ FP8 / INT4-AWQ / NVFP4 to a TRT-LLM engine** (`examples/llm_ptq/`). Its own entry
    already said "pointer, not an example", and its target section `09-TensorRT-LLM` is closed —
    TensorRT-LLM intake was closed 2026-08-28 (above). Sub-8-bit is separately measured as
    unreachable on this stack (§2.2).

---

## 5. EXCLUDE — internal / sensitive (do **not** import)

+ `tools/infer_fuzzing/` — **security**: safety-team ONNX fuzzer; the README leaks a real customer
  model path (Toyota) and internal attack methodology.
+ `tools/memory_usage_safe/` — leaks an internal SSH endpoint / hardcoded IP and an unreleased
  remote QNX timing-server flow.
+ `tools/infer_ref_check_safe/` — undocumented internal debug hooks (`__LUNOWUD`,
  `MYELIN_DUMP_ALL_VALUES`, `MYELIN_SAVE_TENSOR_VALUES`); Safe/LWE runtime.
+ `samples_internal/dlaLoadableExtractor/`, `tools/engine_dumper/`, `tools/plan_converter/` —
  downcast to non-public classes / parse engine-plan binary internals (`api/engine.h`,
  `dispatch/planHeaders.h`).
+ `tools/infer_device*/`, `tools/boot_time_bench/` — internal Myelin / NVRTC / Safe-runtime plumbing.
+ `plugin_internal/` (dlrmBottomMLP, rnRes2*, rnntEncoder, smallTileGEMM) — unreleased plugins.
+ `projects/customer_plugins/{zeekr,edge-llm}/` — **named after customers. Absolutely exclude.**
+ `include/NvInferSerialize.h` — internal `serializeNetwork_INTERNAL()`, marked `@private` and absent
  from the public headers. The cookbook's `utils_network_serialization.py` is an independent
  implementation; do **not** try to align it with this API.
+ All fusion / optimizer / Myelin unit tests (`tutApiFoldReformatIntoMyelin*`, `tutHorizontalMerge*`,
  `tutFuseGELU*`, `tutPointWiseFusion*`, `tutDisableFusion*`, `tutMyelin*`, `tutRaggedTensorLayer*` …)
  — they encode internal pass names, tactic heuristics and nvbug IDs.
+ `documentation/architecture/` (BuilderArch/RuntimeArch + UML) — read it, do not mirror it.
+ `dev_docs/`, `scripts/gitlab_ci/`, `scripts/ai-agents/`, `infrastructure/`, `coverity/`, `capture/`,
  `multigen/`, `plc/`, `optimizer/`, `runtime/`, `samples/README_internal.md` — internal plumbing.
+ Safety samples (`sampleSafeMNIST`, `sampleSafePluginV3`, `trtSafeExec`) — `trtSafeExec`'s own README
  says it is NOT safety-certified and may violate AUTOSAR; the cookbook already has `04-Feature/Safety`.

**Ambiguous, vet first:** `tools/engine_visualizer/` (L7) and `tools/infer_ref_check/` (L9) carry
proprietary SPDX headers and internal URLs/emails — relicense and scrub before adapting.

---

## 6. Reference

### 6.1 Ecosystem repos

| Repo                                                                                                                                                    | Status for the cookbook                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) (rebranded from "TensorRT Model Optimizer", Dec 2025)                               | §3.4                                                                   |
| [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator)                                                                               | **done** → `07-Tool/nvtriPy` (own venv: it brings TensorRT 10 with it) |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)                                                                                           | **no action**, intake closed (§4)                                      |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect), [TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | no action                                                              |
| [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX)                                                                                           | `10-TensorRT-RTX`, R1–R3                                               |
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)                                                                                                   | primary upstream → §3.2                                                |
| [pytorch/TensorRT](https://github.com/pytorch/TensorRT)                                                                                                 | `06-DLFrameworkTRT`, leftovers in §3.6                                 |
| [tensorflow/tensorrt](https://github.com/tensorflow/tensorrt)                                                                                           | **archived Feb 2025**, legacy reference only                           |
