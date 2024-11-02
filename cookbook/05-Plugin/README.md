# 05-Plugin

+ Examples of using TensorRT plugins.

## APIs

+ Example of showing all APIs of plugin.

## APIs

+ Example of showing all APIs of plugin.

## Basic Example

+ Basic example of using `PluginV3` to add a scalar onto the input tensor.

## Basic Example - V2DynamicExt (deprecated)

+ The same as Basic Example, but using `IPluginV2DynamicExt` (deprecated) class.

## Data Dependent Shape

+ Example of using a Data-Dependent-Shape plugin to move all non-zero elements to the left side.

## In-Place Plugin

+ The same as Basic Example, but use in-place plugin (input and output tensor share the same buffer).

## Multi-Version

+ The same as BasicExample, but multiple versions of the plugin are provided to be chose at runtime.

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

## Shape Input Tensor

+ Example of sending a shape input tensor into plugin to reshape another execution tensor by the values of it.

## Tactic+TimingCache

+ The same as BasicExample, but we use our own tactics and timing-cache in the plugin.

## Use cuBLAS

+ Example of using cuBLAS in plugin.

## Use cuBLAS

+ Example of using cuBLAS in plugin.

## UseFP16

+ The same as BasicExample, but enabling FP16 mode.

## UseINT8-PTQ - V2DynamicExt (deprecated)

+ The same as BasicExample, but enabling INT8-PTQ mode with `IPluginV2DynamicExt` (deprecated).

## UseINT8-PTQ

+ The same as BasicExample, but enabling INT8-PTQ mode.
