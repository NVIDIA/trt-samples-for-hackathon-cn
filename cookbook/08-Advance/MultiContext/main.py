# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

n_context = 2
n_test = 10
input_data_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
input_data = {"x": np.load(input_data_file)}

@case_mark
def case_normal(b_skip_memory_copy: bool = False):
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list[:1])  # Remove the output after TopK operator

    # Unpack tw.setup() and rewrite it here
    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
    tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
    tw.n_input = sum([tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tw.tensor_name_list])
    tw.n_output = tw.engine.num_io_tensors - tw.n_input

    context_list = [tw.engine.create_execution_context() for _ in range(n_context)]
    buffer_list = [OrderedDict() for _ in range(n_context)]
    stream_list = [cudart.cudaStreamCreate()[1] for _ in range(n_context)]

    for context, buffer, stream in zip(context_list, buffer_list, stream_list):
        context.set_optimization_profile_async(0, stream)  # We only have 1 optimization-profile, i.e., 0
        for name, data in input_data.items():
            context.set_input_shape(name, data.shape)

        for name in tw.tensor_name_list:
            data_type = tw.engine.get_tensor_dtype(name)
            runtime_shape = context.get_tensor_shape(name)
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            pBuffer = cudart.cudaHostAlloc(n_byte, cudart.cudaHostAllocWriteCombined)[1]
            pointer = ctypes.cast(pBuffer, ctypes.POINTER(ctypes.c_float * trt.volume(runtime_shape)))
            host_buffer = np.ndarray(shape=runtime_shape, buffer=pointer[0], dtype=trt.nptype(data_type))
            device_buffer = cudart.cudaMallocAsync(n_byte, stream)[1]
            buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            for i in range(data.size):
                buffer[name][0].reshape(-1)[i] = data.reshape(-1)[i]

        for name in tw.tensor_name_list:
            context.set_tensor_address(name, buffer[name][1])

    # Do many inference on each context
    for i in range(n_test):
        if not b_skip_memory_copy:
            cudart.cudaMemcpyAsync(buffer_list[0]["x"][1], buffer_list[0]["x"][0], buffer_list[0]["x"][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream_list[0])
            cudart.cudaMemcpyAsync(buffer_list[1]["x"][1], buffer_list[1]["x"][0], buffer_list[1]["x"][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream_list[1])

        context.execute_async_v3(stream_list[0])
        context.execute_async_v3(stream_list[1])

        if not b_skip_memory_copy:
            cudart.cudaMemcpyAsync(buffer_list[0]["y"][0], buffer_list[0]["y"][1], buffer_list[0]["y"][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream_list[0])
            cudart.cudaMemcpyAsync(buffer_list[1]["y"][0], buffer_list[1]["y"][1], buffer_list[1]["y"][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream_list[1])

    for stream in stream_list:
        cudart.cudaStreamSynchronize(stream)

    for buffer in buffer_list:
        cudart.cudaFree(buffer["x"][1])
        cudart.cudaFree(buffer["y"][1])

if __name__ == "__main__":
    case_normal(False)
    case_normal(True)

    print("Finish")
