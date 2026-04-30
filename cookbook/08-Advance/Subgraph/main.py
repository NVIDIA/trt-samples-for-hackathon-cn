# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorrt as trt
import torch
import torch.onnx
from cuda.bindings import runtime as cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

@torch.jit.script
def sum_even(items):
    s = torch.zeros(1, dtype=torch.float32)
    for c in items:
        if c % 2 == 0:
            s += c
    return s

class ExampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, items):
        return sum_even(items)

def build_engine(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    builder_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    input_tensor = network.get_input(0)
    profile.set_shape(input_tensor.name, [1], [4], [64])
    builder_config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")
    return engine_bytes

def run_trt(engine_bytes, items_np):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context")

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, items_np.shape)

    output_shape = tuple(context.get_tensor_shape(output_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output_host = np.empty(output_shape, dtype=output_dtype)

    n_input = items_np.nbytes
    n_output = output_host.nbytes
    d_input = cudart.cudaMalloc(n_input)[1]
    d_output = cudart.cudaMalloc(n_output)[1]

    context.set_tensor_address(input_name, d_input)
    context.set_tensor_address(output_name, d_output)

    cudart.cudaMemcpy(d_input, items_np.ctypes.data, n_input, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_async_v3(0)
    cudart.cudaMemcpy(output_host.ctypes.data, d_output, n_output, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    return output_host

def export_to_onnx():
    items = torch.zeros(4, dtype=torch.float32)
    example = ExampleModel().eval()
    torch.onnx.export(
        example,
        (items, ),
        "example.onnx",
        input_names=["items"],
        output_names=["sum_even"],
        verbose=False,
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"items": {
            0: "N"
        }},
        dynamo=False,
    )

if __name__ == "__main__":
    export_to_onnx()
    engine_bytes = build_engine("example.onnx")

    test_input = np.array([1, 2, 3, 4], dtype=np.float32)
    trt_output = run_trt(engine_bytes, test_input)
    torch_output = ExampleModel().eval()(torch.from_numpy(test_input)).detach().cpu().numpy()

    print("Input:", test_input)
    print("PyTorch output:", torch_output)
    print("TensorRT output:", trt_output)
    print("AllClose:", np.allclose(trt_output, torch_output, atol=1e-6))
