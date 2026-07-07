# 08-Advance

+ Tool combinations of using TensorRT and other CUDA / pyTorch features.

## C++ Static Compilation

+ Static compilation the TensorRT engine into a executable file.

## CUDA graph

+ Use CUDA graph to solve launch bound issue (usually appear in small TensorRT engines).

## MIG

+ Minimal helper example for using TensorRT in Multi-Instance GPU (MIG) mode.

## Multi Context

+ Use multiple execution context to do inference.

## Multi Device

+ Example to show `engine_bytes` can be shared cross devices, but `engine` can not.

## Multi Optimization Profile

+ Use multiple Optimization-Profile to do inference.

## Multi-Stream

+ Use one execution context with multiple CUDA stream.

## Stream and Async

+ Example  to use pinned memory.

##

+ Safety mode is only for Drive Platform (QNX)，https://github.com/NVIDIA/TensorRT/issues/2156

## Subgraph

+ Use cases of parsing ONNX file with subgraph into TensorRT.

##

## Steps to run
