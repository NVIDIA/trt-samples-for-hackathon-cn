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

import os
from pathlib import Path
from time import time_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

data = {"x": np.load(Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy")}
data1 = {"x": np.tile(data["x"], [2, 1, 1, 1])}

@case_mark
def case_normal():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)
    tw.setup(data)

    # Do inference once before CUDA graph capture update internal state
    tw.context.execute_async_v3(0)

    # CUDA Graph capture
    _, stream = cudart.cudaStreamCreate()
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)

    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpyAsync(tw.buffer[name][1], tw.buffer[name][0].ctypes.data, tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    tw.context.execute_async_v3(stream)

    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpyAsync(tw.buffer[name][0].ctypes.data, tw.buffer[name][1], tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    #cudart.cudaStreamSynchronize(stream)  # Do not synchronize during capture
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)

    # CUDA graph launch
    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    # If size of input tensors changes, we need to do inference once, then recapture and launch the CUDA graph.
    """
    tw.setup(data1)
    tw.context.execute_async_v3(0)

    # CUDA Graph capture
    ...
    # CUDA graph launch
    ...
    """

    # Other work after CUDA graph asunch
    for name in tw.tensor_name_list:
        print(name)
        print(tw.buffer[name][0])

    for _, device_buffer, _ in tw.buffer.values():
        cudart.cudaFree(device_buffer)

@case_mark
def case_compare():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)
    tw.setup(data)
    tw.infer(b_print_io=False)

    n_test = 30

    # USe TensorRT directly
    tw.context.execute_async_v3(0)  # Warming up
    cudart.cudaStreamSynchronize(0)

    t0 = time_ns()
    for _ in range(n_test):
        tw.context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    t1 = time_ns()
    print(f"Latency of TensorRT directly     : {(t1-t0)/1000/n_test:.3f}us")

    # Ude TensorRT + CUDA Graph
    _, stream = cudart.cudaStreamCreate()
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cudart.cudaMemcpyAsync(tw.buffer[name][1], tw.buffer[name][0].ctypes.data, tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    tw.context.execute_async_v3(stream)
    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpyAsync(tw.buffer[name][0].ctypes.data, tw.buffer[name][1], tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)

    cudart.cudaGraphLaunch(graphExe, stream)  # Warming up
    cudart.cudaStreamSynchronize(stream)

    t0 = time_ns()
    for _ in range(n_test):
        cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)
    t1 = time_ns()
    print(f"Latency of TensorRT + CUDA-Graph : {(t1-t0)/1000/n_test:.3f}us")

    for _, device_buffer, _ in tw.buffer.values():
        cudart.cudaFree(device_buffer)

if __name__ == "__main__":
    case_normal()  # Basic usage of TensorRT + CUDA-Graph
    case_compare()  # Compare performance between TensorRT and TensorRT + CUDA-Graph

    print("Finish")
