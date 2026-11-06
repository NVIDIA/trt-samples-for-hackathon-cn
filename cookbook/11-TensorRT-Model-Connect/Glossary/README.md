# Glossary

+ Terms you meet in this cookbook, in plain words, with the example that shows each one.

Shape borrowed from `website/docs/getting-started/glossary.md` in
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) (Apache-2.0). Their third
column says "In this project"; ours says **where to go next**, because a cookbook's job is to send
you to the runnable thing.

There is no `main.py` here. This is a reading page.

## Build-time concepts

| Term | Plain meaning | Where in the cookbook |
| --- | --- | --- |
| Network | The graph you describe to TensorRT: layers, tensors, weights. Not yet runnable. | [`02-API/Layer/`](../../02-API/Layer/README.md), [`07-Tool/NetworkPrinter/`](../../07-Tool/NetworkPrinter/README.md) |
| Builder / BuilderConfig | The compiler, and its settings. Everything you can ask of the compiler is a `BuilderConfig` flag. | [`02-API/Builder*/`](../../02-API/README.md) |
| Engine (plan) | The compiled artifact: kernels, schedule, memory plan. Tied to a GPU architecture and a TensorRT version. | [`01-SimpleDemo/`](../../01-SimpleDemo/README.md), [`07-Tool/EnginePrinter/`](../../07-Tool/EnginePrinter/README.md) |
| Optimization profile | The min/opt/max shape range a dynamic engine is compiled for. An engine may hold several. | [`08-Advance/MultiOptimizationProfile/`](../../08-Advance/MultiOptimizationProfile/README.md) |
| Tactic | One candidate kernel for one layer. The builder times tactics and keeps the fastest. | [`04-Feature/TacticSource/`](../../04-Feature/TacticSource/README.md), [`04-Feature/TimingCache/`](../../04-Feature/TimingCache/README.md) |
| Timing cache | Saved tactic timings, so a rebuild skips the measuring. Not portable across GPUs. | [`04-Feature/TimingCache/`](../../04-Feature/TimingCache/README.md) |
| Strongly typed | The network's dtypes are fixed by the graph, not chosen by the builder. The default and the only mode in TensorRT 11. | [`02-API/Layer/Cast/`](../../02-API/Layer/Cast/README.md), skill `trt-strong-typing-migration` |
| Builder optimization level | How hard the builder searches. Higher means slower build, possibly faster engine. | [`04-Feature/BuilderOptimizationLevel/`](../../04-Feature/BuilderOptimizationLevel/README.md) |
| Fusion / Myelin | Several layers compiled into one kernel. Why engine layer names rarely match ONNX node names. | [`07-Tool/EngineVisualization/`](../../07-Tool/EngineVisualization/README.md) |

## Run-time concepts

| Term | Plain meaning | Where in the cookbook |
| --- | --- | --- |
| Runtime | The object that turns engine bytes back into an `ICudaEngine`. | [`01-SimpleDemo/`](../../01-SimpleDemo/README.md) |
| Execution context | One in-flight inference: its shapes, its scratch memory, its bindings. One engine can carry several. | [`08-Advance/MultiContext/`](../../08-Advance/MultiContext/README.md) |
| Binding / IO tensor | A named engine input or output. Bind by the name the **engine** reports, not the name you gave the network. | [`07-Tool/ContextPrinter/`](../../07-Tool/ContextPrinter/README.md) |
| Device memory | The scratch buffer an execution context needs beside the weights. Can be supplied by you. | [`04-Feature/ExecutionContextAllocationStrategy/`](../../04-Feature/ExecutionContextAllocationStrategy/README.md) |
| Output allocator | The callback TensorRT uses when an output's shape is only known once the kernel has run. | [`04-Feature/OutputAllocator/`](../../04-Feature/OutputAllocator/README.md) |
| Data-dependent shape (DDS) | An output whose shape depends on values, not just on input shapes. `NonZero` is the classic one. | [`02-API/Layer/NonZero/`](../../02-API/Layer/NonZero/README.md) |
| CUDA graph | A captured, replayable launch sequence. Bound to the shapes it was captured at. | [`08-Advance/CudaGraph/`](../../08-Advance/CudaGraph/README.md) |
| Weight streaming | Keeping weights in host memory and moving them in as needed, to run a model larger than VRAM. | [`04-Feature/WeightStreaming/`](../../04-Feature/WeightStreaming/README.md) |

## Precision and accuracy

| Term | Plain meaning | Where in the cookbook |
| --- | --- | --- |
| Precision | Which number format a tensor uses: FP32, FP16, BF16, FP8, INT8, INT4. | [`04-Feature/LowBitQuantization/`](../../04-Feature/LowBitQuantization/README.md) |
| Quantization | Representing values in fewer bits with a scale (and sometimes a zero point). | [`04-Feature/OnnxPTQMethod/`](../../04-Feature/OnnxPTQMethod/README.md) |
| Q/DQ | `QuantizeLinear` / `DequantizeLinear` node pairs that tell the builder where quantization boundaries are. | [`02-API/Layer/QDQStructure/`](../../02-API/Layer/QDQStructure/README.md), [`07-Tool/QDQPlacementAutotune/`](../../07-Tool/QDQPlacementAutotune/README.md) |
| Oracle / reference | The trusted answer a TensorRT result is compared against — usually ONNX Runtime or PyTorch. | [`07-Tool/Onnxruntime/`](../../07-Tool/Onnxruntime/README.md), [`07-Tool/CheckTorchOperator/`](../../07-Tool/CheckTorchOperator/README.md) |
| Agreement / cosine / SAD | The statistics `check_array` reports when two tensors disagree. No single one of them is sufficient. | `tensorrt_cookbook.check_array` |

## Extension points

| Term | Plain meaning | Where in the cookbook |
| --- | --- | --- |
| Plugin | A layer you implement yourself, because TensorRT has no equivalent. | [`05-Plugin/`](../../05-Plugin/README.md), [`ParameterizedPlugin/`](../ParameterizedPlugin/README.md) |
| Quick-deployable plugin | A plugin written in Python with `tensorrt.plugin`, no C++ build. Still experimental. | [`05-Plugin/QuickDeployablePlugin/`](../../05-Plugin/QuickDeployablePlugin/README.md) |
| Refit | Replacing an engine's weights without rebuilding it. | [`04-Feature/Refit/`](../../04-Feature/Refit/README.md), [`04-Feature/RefitObserver/`](../../04-Feature/RefitObserver/README.md) |
| Weight stripping | Building an engine without its weights, to ship a small file and refit on arrival. | [`04-Feature/WeightStripping/`](../../04-Feature/WeightStripping/README.md) |
| Version compatibility | An engine that a later TensorRT can load, by carrying a lean runtime inside the plan. | [`04-Feature/VersionCompatibility/`](../../04-Feature/VersionCompatibility/README.md) |
| GPU allocator | Your own implementation of `IGpuAllocator`, installed on the runtime, the builder, or both. | [`04-Feature/GPUAllocator/`](../../04-Feature/GPUAllocator/README.md) |
| Profiler | The `IProfiler` callback that reports per-layer time. | [`04-Feature/Profiler/`](../../04-Feature/Profiler/README.md) |
| Engine inspector | The API that prints an engine's layer information. `trtexec --exportLayerInfo` is a thin wrapper over it. | [`04-Feature/EngineInspector/`](../../04-Feature/EngineInspector/README.md) |

## Things that are easy to confuse

| Pair | The difference |
| --- | --- |
| **Network** vs **engine** | The network is editable and portable; the engine is compiled, opaque, and tied to a GPU architecture and TensorRT version. |
| **Layer** vs **tensor name** | A layer's name does not rename its output tensor in the engine. `GraphTransform` shows a tensor the network calls `Activation1_output` arriving as `(Unnamed Layer* 1) [Activation]_output`. |
| **Optimization profile** vs **execution context** | The profile is a build-time shape range; the context is a run-time instance that selects one profile. |
| **Precision** vs **quantization** | FP16 is a precision change. INT8 with scales is quantization. Changing both at once makes a regression unattributable. |
| **Weight streaming** vs **weight stripping** | Streaming moves weights at run time to fit in VRAM. Stripping omits them from the plan file and refits them later. |
| **TensorRT** vs **TensorRT-RTX** | Different products, different libraries, overlapping APIs. See [`10-TensorRT-RTX/`](../../10-TensorRT-RTX/README.md). |
