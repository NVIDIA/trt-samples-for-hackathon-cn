# 08-Advance

+ Tool combinations of using TensorRT and other CUDA / pyTorch features.

## C++ Static Compilation

+ Static compilation the TensorRT engine into a executable file.

## Context Parallelism

+ Split one attention model across GPUs along the sequence axis, with the collectives inserted into the ONNX graph.

## CUDA graph

+ Use CUDA graph to solve launch bound issue (usually appear in small TensorRT engines).

## Empty Tensor

+ Zero-volume tensors, in the situations where they actually occur.

## Green Context

+ Give a TensorRT engine a fixed slice of one GPU's SMs, from inside the process.

## MIG (Multi-Instance GPU)

+ Why MIG gets a note rather than an example: from inside the process a slice is just a smaller GPU.

## Multi Context

+ Use multiple execution context to do inference.

## Multi Device

+ Example to show `engine_bytes` can be shared cross devices, but `engine` can not.

## Multi Optimization Profile

+ Use multiple Optimization-Profile to do inference.

## Multi-Stream

+ Use one execution context with multiple CUDA stream.

## Multi Task

+ Serve several **different** engines at once, combining threads, streams, CUDA graphs and device pinning.

## Stream and Async

+ Example  to use pinned memory.

## Resource Probe

+ How much GPU memory can you actually use, and in what order must you give it back?

## Safety

+ Safety mode is only for Drive Platform (QNX)，https://github.com/NVIDIA/TensorRT/issues/2156

## Subgraph

+ Use cases of parsing ONNX file with subgraph into TensorRT.

## TensorRT Graph Surgeon

+ Edit a parsed network at the **`INetworkDefinition`** level, between the ONNX parser and the builder.
