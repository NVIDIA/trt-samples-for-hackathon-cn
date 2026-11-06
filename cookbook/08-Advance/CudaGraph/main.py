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

from time import time_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import case_mark, cookbook_path, load_mnist_network_trt, TRTWrapperV1

data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}
data1 = {"x": np.tile(data["x"], [2, 1, 1, 1])}

@case_mark
def case_normal():
    tw = TRTWrapperV1()
    load_mnist_network_trt(tw)
    tw.build()
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
    load_mnist_network_trt(tw)
    tw.build()
    tw.setup(data)
    tw.infer(b_print_io=False)

    n_test = 30

    # Use TensorRT directly
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

@case_mark
def case_graph_per_shape():
    """Capture one CUDA graph per input shape up front, then launch the one that matches.

    A captured graph is a fixed list of launches with fixed pointers and fixed sizes. Change the
    input shape and the graph is wrong, which is why the comment in `case_normal` says to recapture.
    Recapturing per request costs more than it saves; capturing once per shape does not.

    The trick that makes several graphs valid at the same time is the buffer policy: allocate the
    device buffers **once, at the maximum shape, and never reallocate**. A graph bakes in the
    pointers it saw at capture time, so a `cudaFree` + `cudaMalloc` between shapes would leave every
    previously captured graph pointing at freed memory.
    """
    batch_list = [1, 2, 4]  # The profile built by `load_mnist_network_trt` is min 1 / opt 2 / max 4
    max_batch = max(batch_list)

    tw = TRTWrapperV1()
    load_mnist_network_trt(tw)
    tw.build()
    tw.setup({"x": np.tile(data["x"], [max_batch, 1, 1, 1])}, b_print_io=False)

    # One stream, one set of buffers, reused by every graph
    _, stream = cudart.cudaStreamCreate()
    host_buffer = {name: tw.buffer[name][0] for name in tw.tensor_name_list}
    device_buffer = {name: tw.buffer[name][1] for name in tw.tensor_name_list}
    item_size = {name: tw.buffer[name][0].itemsize for name in tw.tensor_name_list}
    for name in tw.tensor_name_list:
        tw.context.set_tensor_address(name, device_buffer[name])

    def bytes_for(name, batch):
        """Bytes of the first `batch` rows -- the per-shape copy size baked into each graph."""
        shape = host_buffer[name].shape
        return int(np.prod(shape[1:])) * batch * item_size[name]

    graph_of_batch = {}
    for batch in batch_list:
        tw.context.set_input_shape("x", [batch, 1, 28, 28])
        tw.context.execute_async_v3(stream)  # Update internal state before capturing
        cudart.cudaStreamSynchronize(stream)

        cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for name in tw.tensor_name_list:
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpyAsync(device_buffer[name], host_buffer[name].ctypes.data, bytes_for(name, batch), cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        tw.context.execute_async_v3(stream)
        for name in tw.tensor_name_list:
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpyAsync(host_buffer[name].ctypes.data, device_buffer[name], bytes_for(name, batch), cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        _, graph = cudart.cudaStreamEndCapture(stream)
        _, graph_exe = cudart.cudaGraphInstantiate(graph, 0)
        graph_of_batch[batch] = graph_exe
        print(f"    Captured a graph for batch {batch}")

    # Reference values, computed the ordinary way, for every batch we captured
    reference = {}
    for batch in batch_list:
        host_buffer["x"][:batch] = np.tile(data["x"], [batch, 1, 1, 1])
        tw.context.set_input_shape("x", [batch, 1, 28, 28])
        for name in tw.tensor_name_list:
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpyAsync(device_buffer[name], host_buffer[name].ctypes.data, bytes_for(name, batch), cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        tw.context.execute_async_v3(stream)
        for name in tw.tensor_name_list:
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpyAsync(host_buffer[name].ctypes.data, device_buffer[name], bytes_for(name, batch), cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        reference[batch] = {name: host_buffer[name][:batch].copy() for name in tw.tensor_name_list if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    # A request sequence with a shape that changes every call: no recapture, just a lookup
    for batch in [2, 1, 4, 2, 4, 1]:
        host_buffer["x"][:batch] = np.tile(data["x"], [batch, 1, 1, 1])
        cudart.cudaGraphLaunch(graph_of_batch[batch], stream)
        cudart.cudaStreamSynchronize(stream)
        for name, expected in reference[batch].items():
            got = host_buffer[name][:batch]
            assert np.allclose(got, expected, atol=1e-4), f"batch {batch}, tensor {name}"
    print(f"    Ran [2, 1, 4, 2, 4, 1] against {len(graph_of_batch)} cached graphs, every result matched its reference")

    # The failure this guards against: a graph replays the shape it was captured at, and nothing
    # checks that the shape is still the one you want.
    output_name = [name for name in tw.tensor_name_list if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT][0]
    host_buffer[output_name][:] = -1.0  # Poison, so stale rows are visible
    host_buffer["x"][:max_batch] = np.tile(data["x"], [max_batch, 1, 1, 1])
    cudart.cudaGraphLaunch(graph_of_batch[1], stream)  # Wrong graph on purpose
    cudart.cudaStreamSynchronize(stream)
    n_written = int(np.sum(~np.all(host_buffer[output_name] == -1.0, axis=tuple(range(1, host_buffer[output_name].ndim)))))
    print(f"    Launched the batch-1 graph with {max_batch} rows of input: {n_written} of {max_batch} output rows were written")
    assert n_written == 1, "The batch-1 graph must still only write one row"
    print("    The graph obeys its capture, not your intent -- which is why the lookup above has to be right")

    for graph_exe in graph_of_batch.values():
        cudart.cudaGraphExecDestroy(graph_exe)
    cudart.cudaStreamDestroy(stream)
    for _, buffer, _ in tw.buffer.values():
        cudart.cudaFree(buffer)

if __name__ == "__main__":
    case_normal()  # Basic usage of TensorRT + CUDA-Graph
    case_compare()  # Compare performance between TensorRT and TensorRT + CUDA-Graph
    case_graph_per_shape()  # Capture one graph per input shape, then dispatch by shape

    print("Finish")
