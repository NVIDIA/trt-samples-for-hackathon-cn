# 07-Tool

+ Tools of using TensorRT beyond the original APIs.

## Check pyTorch Operator

+ A template to check whether a operator in pyTorch can be used in TensorRT.

## Context Printer

+ Print input / output shape information for the current context.

## Debug Utils

+ Runtime and debug helpers from `tensorrt_cookbook`, exercised in one place.

## Engine Printer

+ Print the header, device requirements and I/O tables of a TensorRT engine, from the plan file alone.

## Engine Visualization

+ One engine layer-info JSON, five outputs: Graphviz DOT and the picture it renders, a

## FP16 Tuning

+ A tool to fine-tune precision and performance of a engine, by trying to pull back the precision of layers from FP16 to FP32.

## List APIs

+ List all the APIs in TensorRT package.

## MPI Utils

+ Complete MPI utility example based on `tensorrt_cookbook` wrappers.

## Netron

+ A visualization tool for neural-network graphs, including ONNX and many other formats.

## Network Printer

+ Print information of layers and tensors in the network.

## Network Serialization and Deserialization

+ Serialize a network into a json file, and deserialize it back into a INetwork.

## Nsight Deep Learning Designer

+ An integrated development environment that helps developers efficiently design and optimize deep neural networks for high-performance inference.

## Nsight Systems

+ Program performance analysis tool (replacing the old performance analysis tools nvprof and nvvp).

## Onnx

+ An open source format for AI models, both deep learning and traditional ML.

## ONNX FP8 Q/DQ Convert

+ Rewrite Transformer-Engine's custom FP8 Q/DQ operators into standard opset-19 `QuantizeLinear` / `DequantizeLinear`.

## Onnx Graphsurgeon

+ A python library for ONNX compute graph edition, which different from the library *onnx*.

## OnnxVisualization

+ A tool to fold the repeated sub-graphs of a flat ONNX into shared local functions, making the model more readable in Netron.

## Onnx Weight Separator

+ A tool to separate weights from a ONNX file, or compose weights to a ONNX file, usually for visualization of a remote large ONNX file.

## Onnx Runtime

+ A cross-platform inference engine for ONNX models, and the reference a TensorRT result is checked against.

## Polygraphy - Client tool

+ CLI tool of polygraphy (deep learning model debugger).

## QDQPlacementAutotune

+ ModelOptimizer's Q/DQ placement search, driven by **real TensorRT latency** rather than by a proxy.

## TritonServerDeploy

+ Deploy a TensorRT engine on Triton Inference Server, end to end.

## nvtripy

+ An eager-mode Python frontend for TensorRT, installed into its own virtual environment.

## nvtx

+ Use NVIDIA®Tools Extension SDK to add mark in timeline of Nsight systems.

## trex - TensorRT Engine Explorer

+ Explore the structure and performance of a **built** engine, from the JSON that `trtexec` exports.

## trtexec

+ Command-line tool of TensorRT, attached with an end-to-end performance test tool.
