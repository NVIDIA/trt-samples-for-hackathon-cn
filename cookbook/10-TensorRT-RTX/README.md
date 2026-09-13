# 10-TensorRT-RTX

+ TensorRT-RTX basic examples and TensorRT-vs-RTX API-diff examples.

> **Optional package.** This example needs `tensorrt_rtx`, which is **not** part of the base
> environment. Install it with `pip install tensorrt_rtx` (it pulls `tensorrt_rtx_cu13{,_libs,_bindings}`
> and coexists with `tensorrt` — they are separate Python modules and neither shadows the other).
> Without it every case here prints `[SKIP] tensorrt_rtx is not installed` and exits 0.

+ This folder keeps the original basic demos (`main.py`, `compare.py`) and adds a focused set of examples for APIs that are present in TensorRT-RTX (1.4.0.76) but not in TensorRT (10.16.2.5).

## RTX-only API summary (vs TensorRT 10.16.2.5)

+ Detailed diff list: `API_DIFF_TRT_vs_TRT_RTX.md`

+ New enums / types:
	+ `ComputeCapability`
	+ `CudaGraphStrategy`
	+ `DynamicShapesKernelSpecializationStrategy`
	+ `EngineValidity`
	+ `EngineInvalidityDiagnostics`
	+ `IRuntimeCache`

+ New/extended members:
	+ `BuilderFlag.REQUIRE_USER_ALLOCATION`
	+ `IBuilderConfig.{set_compute_capability, get_compute_capability, num_compute_capabilities}`
	+ `IExecutionContext.is_stream_capturable(...)`
	+ `IRuntimeConfig.{create_runtime_cache, set_runtime_cache, get_runtime_cache}`
	+ `IRuntimeConfig.{cuda_graph_strategy, dynamic_shapes_kernel_specialization_strategy}`
	+ `Runtime.{engine_header_size, get_engine_validity(...)}`

## Examples

```bash
# Original demos
python3 main.py
python3 compare.py

# RTX-only APIs: all-in-one
python3 00-AllInOne/main.py

# RTX-only APIs: one feature per script
python3 01-ComputeCapability/main.py
python3 02-RuntimeEngineValidity/main.py
python3 03-RuntimeCache/main.py
python3 04-RuntimeConfigStrategy/main.py
python3 05-ExecutionContextStreamCapturable/main.py
python3 06-BuilderRequireUserAllocation/main.py
python3 07-Enums/main.py

# The deploy side as a workflow, measured
python3 08-DeployTimeRuntimeConfig/main.py
```

+ [`08-DeployTimeRuntimeConfig/`](08-DeployTimeRuntimeConfig/README.md) is the only one with its own
  README, because it reports measurements rather than API calls: the runtime cache is worth
  **17.7x** on the first inference in a fresh process (and the CUDA driver's own `~/.nv/ComputeCache`
  has to be pinned before that number means anything), `EAGER` kernel specialization is **2.8x
  slower** than `NONE` at an unseen shape, and `WHOLE_GRAPH_CAPTURE` changes nothing on a
  compute-bound engine. It also covers ahead-of-time compute-capability targeting -- including the
  trap that `num_compute_capabilities` must be **assigned** before `set_compute_capability` will do
  anything, which otherwise returns `False` and silently builds for the current device -- and records
  which RTX features do **not** work on this B200: `REFIT` / `STRIP_PLAN` cannot build at all, and
  `get_engine_validity` rejects plans it then runs. TensorRT-RTX enumerates SM75-SM121, this is SM100.
