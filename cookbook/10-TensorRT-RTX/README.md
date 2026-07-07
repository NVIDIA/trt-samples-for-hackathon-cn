# 10-TensorRT-RTX

+ TensorRT-RTX basic examples and TensorRT-vs-RTX API-diff examples.

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
```
