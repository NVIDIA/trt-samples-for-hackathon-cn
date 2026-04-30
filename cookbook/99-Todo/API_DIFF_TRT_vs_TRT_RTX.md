# API Diff: TensorRT 10.16.2.5 vs TensorRT-RTX 1.4.0.76

Comparison source files:
- `cookbook/07-Tool/ListAPIs/output/result-tensorrt-10.16.2.5.log`
- `cookbook/07-Tool/ListAPIs/output/result-tensorrt_rtx-1.4.0.76.log`

Filtered to APIs that are present in TensorRT-RTX and absent in TensorRT.

## RTX-only members (including enum entries)

- `BuilderFlag.REQUIRE_USER_ALLOCATION`

- `ComputeCapability.{NONE, CURRENT, SM75, SM80, SM86, SM89, SM120}`
- `CudaGraphStrategy.{DISABLED, WHOLE_GRAPH_CAPTURE}`
- `DynamicShapesKernelSpecializationStrategy.{LAZY, EAGER, NONE}`
- `EngineValidity.{VALID, SUBOPTIMAL, INVALID}`
- `EngineInvalidityDiagnostics.{VERSION_MISMATCH, UNSUPPORTED_CC, OLD_CUDA_DRIVER, OLD_CUDA_RUNTIME, INSUFFICIENT_GPU_MEMORY, MALFORMED_ENGINE, CUDA_ERROR}`

- `IBuilderConfig.set_compute_capability(...)`
- `IBuilderConfig.get_compute_capability(...)`
- `IBuilderConfig.num_compute_capabilities`

- `IExecutionContext.is_stream_capturable(...)`

- `IRuntimeCache.serialize(...)`
- `IRuntimeCache.deserialize(...)`
- `IRuntimeCache.reset(...)`

- `IRuntimeConfig.create_runtime_cache(...)`
- `IRuntimeConfig.get_runtime_cache(...)`
- `IRuntimeConfig.set_runtime_cache(...)`
- `IRuntimeConfig.cuda_graph_strategy`
- `IRuntimeConfig.dynamic_shapes_kernel_specialization_strategy`

- `Runtime.engine_header_size`
- `Runtime.get_engine_validity(...)`
