# Torch-TensorRT

+ Use Torch-TensorRT (PyTorch 2.x Dynamo path) to compile a torch network and run it.
+ The sample also includes a `torch.compile` path and prints latency comparison between the two.

+ Steps to run.

```bash
python3 main.py
```

## Sub-examples

| Directory | Topic |
| --- | --- |
| [`EngineCaching/`](EngineCaching/README.md) | Reuse built engines across compilations, sessions and weight changes |
| [`DynamicShapes/`](DynamicShapes/README.md) | One engine for a range of input shapes; AOT vs JIT out of range |
| [`SaveLoad/`](SaveLoad/README.md) | Persist a compiled module and keep its dynamic shapes |
| [`TorchCompileBackend/`](TorchCompileBackend/README.md) | AOT (`ir="dynamo"`) vs JIT (`torch.compile`), and how to choose |
| [`CudaGraphs/`](CudaGraphs/README.md) | Replay as a CUDA graph, and when it is worth anything |
| [`MixedPrecisionAutocast/`](MixedPrecisionAutocast/README.md) | FP16/BF16 on a strongly typed TensorRT 11 network |
| [`ConverterOverloading/`](ConverterOverloading/README.md) | Replace the lowering of one operator |
| [`CustomKernelPlugin/`](CustomKernelPlugin/README.md) | Run a Triton kernel inside the engine via a QDP plugin |
| [`MutableModule/`](MutableModule/README.md) | A compiled module that follows its source weights |
| [`Refit/`](Refit/README.md) | Swap weights into a built engine without recompiling (14.7x faster) -- and the graph mismatch it accepts silently |
| [`WeightStreaming/`](WeightStreaming/README.md) | Run a model whose weights do not fit, by choosing a device budget |
| [`DistributedInference/`](DistributedInference/README.md) | Data-parallel across GPUs (3.1x on 4); why tensor-parallel is blocked here |

```bash
cd $TRT_COOKBOOK_PATH
python3 tests/run_tests.py --tags torch-tensorrt
```

---

# Things that succeed without doing anything

Every sub-example above was written by running the upstream tutorial, measuring
what it claimed, and keeping whatever disagreed. The same failure mode came back
nine times: **the call succeeds, the results are correct, and the thing you asked
for did not happen.** Nothing raises, nothing warns, and the only evidence is a
number you have to go and look for.

They are collected here because the individual READMEs cannot warn you about a
trap you do not yet know exists.

## A. Silently does nothing

Observed, not theorised. In each row the code ran, returned correct output, and
had no effect.

| What looks fine | What actually happened | How it was caught | Where |
| --- | --- | --- | --- |
| `enabled_precisions={torch.float16}` | **Nothing.** TensorRT 11 networks are strongly typed, so the builder has no type to choose. Output stays FP32, latency unchanged, error exactly `0.00e+00`, **no warning** | Output dtype was still `float32` | [`MixedPrecisionAutocast/`](MixedPrecisionAutocast/README.md) |
| `@dynamo_tensorrt_converter(..., priority=STANDARD)` overriding a built-in op | **Nothing.** `STANDARD` appends to the candidate list; the built-in converter is consulted first and wins. Your converter is dead code | A call counter inside the converter read `0` | [`ConverterOverloading/`](ConverterOverloading/README.md) |
| `autocast_excluded_nodes={"^linear$"}` | **Nothing.** Patterns match the *lowered* graph's node names, not the ones `torch.export` prints. A pattern that matches nothing is ignored | Error was identical to no exclusion at all | [`MixedPrecisionAutocast/`](MixedPrecisionAutocast/README.md) |
| `torch_tensorrt.compile(...)` on a small model | **No engine built.** `min_block_size` defaults to 5; a sub-graph with fewer operators stays in eager PyTorch | Counted `call_module` nodes: zero `_run_on_acc_*` | [`DynamicShapes/`](DynamicShapes/README.md) |
| `torch_tensorrt.save(m, path, arg_inputs=[example])` on a dynamic model | **Model frozen static.** `retrace` defaults to `True` and `dynamic_shapes` to `None`, so the re-export specializes on the example tensor's batch | Loaded model failed `Guard failed: x.size()[0] == 4` at every other batch | [`SaveLoad/`](SaveLoad/README.md) |
| Mutating a model *after* `torch_tensorrt.compile` | **Compiled module unchanged.** The engine holds a snapshot; it now disagrees with eager by `2.495` and says nothing | Compared the compiled module against eager after the change | [`MutableModule/`](MutableModule/README.md) |

## B. Works, but the cost goes the other way

Not failures — measurements that contradict the obvious expectation, or the
upstream documentation.

| Expectation | Measured | Why | Where |
| --- | --- | --- | --- |
| A QDP plugin removes the graph break, so it is faster | 3 segments → **1 segment**, and **2.51x slower** (0.281 → 0.706 ms) | The generated plugin is **JIT**: TensorRT calls back into Python on every inference. That costs more than the graph break it replaced. AOT plugins are the fix | [`CustomKernelPlugin/`](CustomKernelPlugin/README.md) |
| CUDA graphs speed up inference | **1.02x** on ResNet18 batch 16 — nothing. 1.97x on 40 small `Linear` layers | Only *launch* overhead is removed. TensorRT already fused ResNet into a few long kernels. It is the kernel count that decides, not the batch size | [`CudaGraphs/`](CudaGraphs/README.md) |
| A recorded CUDA graph is reused per shape | Only **one** recording is kept. Returning to a previously seen shape re-records anyway (0.664 ms vs 0.230 ms replay) | A workload alternating between two shapes is better off with CUDA graphs disabled | [`CudaGraphs/`](CudaGraphs/README.md) |
| Enabling the engine cache makes the first build faster | The *cold-cache* build is slightly **slower** — the engine has to be serialised and written | The win arrives on the third compilation, not the second | [`EngineCaching/`](EngineCaching/README.md) |

## C. Fails loudly (listed so the loud ones are not confused with the quiet ones)

| Situation | Behaviour |
| --- | --- |
| Data-dependent control flow (`if y.sum() > 0`) with `ir="dynamo"` | `torch.export` raises `GuardOnDataDependentSymNode`. The whole AOT path is unavailable — not slower, unavailable. JIT handles it by splitting into 2 frames ([`TorchCompileBackend/`](TorchCompileBackend/README.md)) |
| Input outside the declared shape range, AOT | `RuntimeError` from `setInputShape`; the optimization profile is a hard bound. JIT recompiles instead (~7 s) and continues ([`DynamicShapes/`](DynamicShapes/README.md)) |
| Saving a `torch.compile` result | `ValueError: Saving nn.Module directly is not supported`. There is no compiled artifact to serialise ([`TorchCompileBackend/`](TorchCompileBackend/README.md)) |

## What to check, every time

The traps in section A share one shape: a boolean or a string was accepted and
then ignored. Three habits catch all of them.

1. **Count the engines.** `sum(1 for n in module.graph.nodes if n.op == "call_module")`
   — zero means nothing was compiled, and every latency number taken against that
   module is meaningless. This alone catches the `min_block_size` trap and is
   printed by several sub-examples for that reason.
2. **Compare against eager PyTorch, not against another TensorRT run.** A
   converter that reproduces TensorRT's own answer proves much less than one that
   reproduces PyTorch's; and a refit that never happened only shows up against
   the *new* weights' eager output.
3. **Exercise the thing you asked for, not the thing you traced with.** A dynamic
   model has to be run at several shapes; a refit has to be run after the weights
   change; a custom converter needs a counter. Running only the example input
   passes in every row of section A.

Where a sub-example asserts one of these, the assertion is the point of the case —
not decoration.

## Note on this directory's `main.py`

It passes no `enabled_precisions` and no `truncate_long_and_double`. Both belong
to the weakly typed builder of TensorRT 10 and earlier; on TensorRT 11 the first
is the silent no-op documented above.
