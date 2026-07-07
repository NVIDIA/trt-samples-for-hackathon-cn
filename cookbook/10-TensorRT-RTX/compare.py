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

from collections import OrderedDict
from time import time_ns

import numpy as np
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import build_mnist_network_trt, cookbook_path

onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}
n_test = 200
n_warmup = 20

def benchmark(b_is_trt_rtx: bool):

    if b_is_trt_rtx:
        import tensorrt_rtx as trt
        package_suffix = "-RTX"
    else:
        import tensorrt as trt
        package_suffix = "    "

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    network = builder.create_network()
    profile = builder.create_optimization_profile()

    output_tensor_list = build_mnist_network_trt(builder_config=builder_config, network=network, profile=profile)
    # We do not use `load_mnist_network_trt` here since it calls function `trt.OnnxParser` inside

    for tensor in output_tensor_list:
        network.mark_output(tensor)
    engine_bytes = builder.build_serialized_network(network, builder_config)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    context.set_input_shape("x", data["x"].shape)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    buffer = OrderedDict()
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(n_byte)[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]

    for name, value in data.items():
        buffer[name][0] = np.ascontiguousarray(value)

    for name in tensor_name_list:
        context.set_tensor_address(name, buffer[name][1])

    # Run once
    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpy(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_async_v3(0)
    for name in tensor_name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaStreamSynchronize(0)

    # Warm up and test
    for _ in range(n_warmup):
        context.execute_async_v3(0)

    t0 = time_ns()
    for _ in range(n_test):
        context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    t1 = time_ns()
    print(f"TensorRT{package_suffix} latency: {(t1 - t0) / 1_000 / n_test:.3f} us")

    for _, device_buffer, _ in buffer.values():
        cudart.cudaFree(device_buffer)

    return

if __name__ == "__main__":

    benchmark(False)
    benchmark(True)

    print("Finish")
