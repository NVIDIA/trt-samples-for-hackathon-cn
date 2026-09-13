# 04-Feature

+ Examples of the feature APIs, which are not necessary in a basic workflow.

## Aux Stream

+ Use auxiliary streams for infernece.

## Builder Optimization Level

+ Set optimization level for TRT building.

## Engine Inspector

+ Steps to run.

## Corner Case

+ UINT8, BOOL, NaN and fan-out: the semantics that only show up when the types are unusual.

## DLA Standalone

+ Demonstrates TensorRT DLA standalone build configuration.

## Use PluginV2DynamicExt

+ Data type and data format supported in TensorRT.

## Debug Tensor

+ Steps to run.

## Empty Tensor

+ Use case of empty tensors.

## Engine Inspector

+ Steps to run.

## Enumerate

+ Discover all enum classes in the current TensorRT package and print their members.

## ErrorRecorder

+ Steps to run.

## Event

+ Demonstrates `IExecutionContext.set_input_consumed_event` and `IExecutionContext.get_input_consumed_event`.

## Execution Context Allocation Strategy

+ Steps to run

## GPU Allocator

+ This example shows how to attach custom allocator to `Runtime` / `ExecutionContext`.

## Hardware Compatibility

+ Steps to run.

## Labeled Dimension

+ Steps to run

## Lean and Dispatch Rutnime

+ Use Lean and Dispatch Rutnime to do inference.

## Logger

+ Steps to run.

## Low-Bit Quantization

+ NVFP4, MXFP8 and INT4-AWQ from PyTorch: what the formats cost, and which of them actually reach a TensorRT engine.

## ONNX PTQ Method

+ Calibration **method** on an ONNX model — entropy vs max, per-node calibration, and INT4 weight-only.

## Output Allocator

+ Write an `IOutputAllocator` for a detection head, where the number of output boxes is decided by the picture.

## Profiler

+ `IProfiler` and, on top of it, a latency report an application can produce for itself.

## Profiling Verbosity

+ Steps to run.

## Progress Monitor

+ Steps to run.

## Refit

+ Steps to run.

## RefitObserver

+ `IRefitterObserver` — record at build time **how** every refittable engine weight is produced from

## Safety mode

+ Safety mode is only available on NVIDIA Drive platforms (QNX).

## Serialization Config

+ Refer to `02-API/CudaEngine`.

## Sparsity

+ Example of enabling sparse weights in TensorRT to reduce compute cost on supported hardware.

## Tactic Source

+ Choose which kernel libraries (cuBLAS, cuDNN, cuBLASLt, edge mask convolutions) the builder may draw tactics from.

## Timing Cache

+ Usage of timing cache to reduce engine building time, including editable timing cache.

## Version Compatibility

+ Steps to run.

## Weight Streaming

+ Steps to run.

## WeightStripping

+ Build a weight-stripped engine with `BuilderFlag.STRIP_PLAN` and refit it back to a full one from the original ONNX.
