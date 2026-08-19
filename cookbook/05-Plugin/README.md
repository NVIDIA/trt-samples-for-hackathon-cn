# 05-Plugin

+ Examples of using TensorRT plugins.

## APIs

+ Example of showing all APIs of plugin.

## AliasedIOPlugin

+ A Python plugin that writes into one of its own inputs, using the aliased-I/O capability of `IPluginV3OneBuildV2`.

## Basic Example

+ Basic example of using `PluginV3` to add a scalar onto the input tensor.

## Basic Example - V2DynamicExt (deprecated)

+ The same as Basic Example, but use `IPluginV2DynamicExt` class (deprecated).

## Basic Example - static register (deprecated)

+ The same as Basic Example, but register the plugin in a static way (deprecated).

## CuteDSLPlugin

+ An `IPluginV3` whose kernel is written in **CuteDSL**, CUTLASS's Python DSL.

## Data Dependent Shape

+ Example of using a Data-Dependent-Shape plugin to move all non-zero elements to the left side.

## INT8-QDQ-Plugin

+ Minimal example combining QDQ layers with a plugin insertion point.

## Identity plugin

+ Basic example of using `PluginV3` to copy input to output.

## In-Place Plugin

+ The same as Basic Example, but use in-place plugin (input and output tensor share the same buffer).

## MigrationV2toV3

+ Migrate a Python plugin from the deprecated `IPluginV2DynamicExt` to `IPluginV3`.

## Multi-Version

+ The same as BasicExample, but multiple versions of the plugin are provided to be chose at runtime.

## NcclPlugin

+ Minimal TensorRT `PluginV3` + NCCL `send/recv` example.

## ONNX PTQ With Plugin

+ Quantize an ONNX graph that contains a **custom plugin op**, then build and run it.

## ONNX Parser and Plugin

+ Example of combinating the usage of model from ONNX and plugin.

## Pass Host Data

+ Example of passing a host pointer (pointing to anything like array, structure or even nullptr) into plugin at runtime.

## Plugin Inside Engine - C++

+ Example of serializing a plugin inside a TensorRT engine (no `.so` needed at runtime) using C++ APIs.

## Plugin Inside Engine - Python

+ Example of serializing a plugin inside a TensorRT engine (no `.so` needed at runtime) using C++ APIs.

## PythonPlugin

+ The same as BasicExample, but we make the workflow totally in Python script.

## Quick Deployable Python plugin

+ The same as BasicExample, but use decorated functions to simplift the workflow in Python plugin.

## Resource

+ Example of using TensorRT `IPluginResource` to share information between two `PluginV3` layers in one network.

## Shape Input Tensor

+ Example of sending a shape input tensor into plugin to reshape another execution tensor by the values of it.

## Tactic+TimingCache

+ The same as BasicExample, but we use our own tactics and timing-cache in the plugin.

## Triton AOT Plugin

+ Compile an **OpenAI-Triton** kernel *ahead of time* and ship it inside a C++ `IPluginV3`.

## Use cuBLAS

+ Example of using cuBLAS in plugin.

## Use cuFFT

+ Call cuFFT from a plugin, and give the ONNX `DFT` operator somewhere to go.

## UseFP16

+ The same as BasicExample, but enabling FP16 mode.
