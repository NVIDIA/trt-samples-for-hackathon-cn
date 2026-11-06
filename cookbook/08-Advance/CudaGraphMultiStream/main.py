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

from threading import Barrier, Thread
from time import time_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import TRTWrapperV1, case_mark

n_thread_list = [1, 2, 4]
n_warm = 50
n_test = 2000
n_repeat = 5  # Contention only ever makes a sample slower, so the best repeat is the honest one
shape = [1, 16, 32, 32]

def build_small_engine():
    """A small engine, where issuing the inference costs more CPU time than the GPU work it starts.

    Do not try to make this "launch bound" by stacking many small layers: the builder fuses any
    chain of pointwise operations into a single kernel, so 24 layers and 3 layers both come out as
    one. Removing per-launch CPU cost is what a graph does, and a small engine is where that cost
    is the ceiling - which is the same regime `08-Advance/CudaGraph` describes.
    """
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("x", trt.float32, shape)
    tw.profile.set_shape(tensor.name, shape, shape, shape)
    for activation_type in [trt.ActivationType.RELU, trt.ActivationType.SIGMOID, trt.ActivationType.TANH]:
        tensor = tw.network.add_activation(tensor, activation_type).get_output(0)
    tensor.name = "y"
    tw.build([tensor])
    tw.setup({"x": np.ascontiguousarray(np.random.rand(*shape).astype(np.float32))}, b_print_io=False)
    return tw

class Worker:
    """One thread's private state: its own context, stream and device buffers.

    An `IExecutionContext` holds the scratch space for one inference in flight, so two threads
    enqueueing into it at once corrupt each other. The buffers are allocated once and never freed,
    because a captured graph replays the addresses it saw at capture time.
    """

    def __init__(self, engine, host_input):
        self.engine = engine
        self.context = engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()
        self.host_input = host_input
        self.buffer = {}
        for name in [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]:
            n_byte = int(trt.volume(self.context.get_tensor_shape(name)) * np.dtype(trt.nptype(engine.get_tensor_dtype(name))).itemsize)
            _, address = cudart.cudaMalloc(n_byte)
            self.buffer[name] = (address, n_byte)
            self.context.set_tensor_address(name, address)
        self.graph_exe = None

    def enqueue_once(self):
        address, n_byte = self.buffer["x"]
        cudart.cudaMemcpyAsync(address, self.host_input.ctypes.data, n_byte, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.execute_async_v3(self.stream)

    def capture(self):
        # The first enqueue after the shapes are set does one-off internal work that is illegal
        # inside a capture, so it has to happen before `cudaStreamBeginCapture`, not inside it.
        self.enqueue_once()
        cudart.cudaStreamSynchronize(self.stream)
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)
        self.enqueue_once()
        _, graph = cudart.cudaStreamEndCapture(self.stream)
        _, self.graph_exe = cudart.cudaGraphInstantiate(graph, 0)

    def run(self, n, b_graph):
        for _ in range(n):
            if b_graph:
                cudart.cudaGraphLaunch(self.graph_exe, self.stream)
            else:
                self.enqueue_once()
        cudart.cudaStreamSynchronize(self.stream)

    def free(self):
        if self.graph_exe is not None:
            cudart.cudaGraphExecDestroy(self.graph_exe)
        cudart.cudaStreamDestroy(self.stream)
        for address, _ in self.buffer.values():
            cudart.cudaFree(address)

def measure(engine, host_input, n_thread, b_graph):
    """Throughput of `n_thread` threads each driving its own context, in inferences per second."""
    worker_list = [Worker(engine, host_input) for _ in range(n_thread)]
    if b_graph:
        for worker in worker_list:
            worker.capture()

    barrier = Barrier(n_thread + 1)  # Start the clock only once every thread is ready to launch

    def body(worker):
        worker.run(n_warm, b_graph)
        barrier.wait()
        worker.run(n_test, b_graph)
        barrier.wait()

    thread_list = [Thread(target=body, args=(worker, )) for worker in worker_list]
    for thread in thread_list:
        thread.start()
    barrier.wait()
    t0 = time_ns()
    barrier.wait()
    t1 = time_ns()
    for thread in thread_list:
        thread.join()
    for worker in worker_list:
        worker.free()
    return n_thread * n_test / ((t1 - t0) / 1e9)

@case_mark
def case_scaling():
    tw = build_small_engine()
    host_input = np.ascontiguousarray(np.random.rand(*shape).astype(np.float32))
    print(f"    {n_test} inferences per thread, best of {n_repeat} repeats")
    print(f"    {'threads':>8} | {'no graph':>14} | {'CUDA graph':>14} | {'speedup':>8}")
    for n_thread in n_thread_list:
        qps_plain = max(measure(tw.engine, host_input, n_thread, False) for _ in range(n_repeat))
        qps_graph = max(measure(tw.engine, host_input, n_thread, True) for _ in range(n_repeat))
        print(f"    {n_thread:>8} | {qps_plain:>11.1f}/s | {qps_graph:>11.1f}/s | {qps_graph / qps_plain:>7.2f}x")

@case_mark
def case_context_per_thread():
    """Two threads sharing one context corrupt each other; one context each does not."""
    tw = build_small_engine()
    host_input = np.ascontiguousarray(np.random.rand(*shape).astype(np.float32))

    reference = Worker(tw.engine, host_input)
    reference.enqueue_once()
    cudart.cudaStreamSynchronize(reference.stream)
    expected = np.empty(shape, dtype=np.float32)
    address, n_byte = reference.buffer["y"]
    cudart.cudaMemcpy(expected.ctypes.data, address, n_byte, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    reference.free()

    worker_list = [Worker(tw.engine, host_input) for _ in range(2)]
    thread_list = [Thread(target=worker.run, args=(n_test, False)) for worker in worker_list]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    for i, worker in enumerate(worker_list):
        got = np.empty(shape, dtype=np.float32)
        address, n_byte = worker.buffer["y"]
        cudart.cudaMemcpy(got.ctypes.data, address, n_byte, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        assert np.allclose(got, expected, atol=1e-4), f"Worker {i} disagrees with the single-threaded reference"
        worker.free()
    print(f"    {len(thread_list)} threads, one context each: both outputs match the single-threaded reference")
    print("    Sharing one context across these threads is what the TensorRT API forbids -- the")
    print("    scratch space inside a context belongs to one inference in flight, not to a pool")

@case_mark
def case_graph_replays_addresses():
    """A graph replays the pointers it captured, so each thread needs its own buffers."""
    tw = build_small_engine()
    host_input = np.ascontiguousarray(np.random.rand(*shape).astype(np.float32))
    worker = Worker(tw.engine, host_input)
    worker.capture()

    worker.run(1, True)
    first = np.empty(shape, dtype=np.float32)
    address, n_byte = worker.buffer["y"]
    cudart.cudaMemcpy(first.ctypes.data, address, n_byte, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # New data written into the *same* host buffer is picked up, because the captured
    # `cudaMemcpyAsync` reads from the address it was given, not from a snapshot of the data.
    host_input[:] = np.ascontiguousarray(np.random.rand(*shape).astype(np.float32))
    worker.run(1, True)
    second = np.empty(shape, dtype=np.float32)
    cudart.cudaMemcpy(second.ctypes.data, address, n_byte, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assert not np.allclose(first, second), "Replaying after new input should change the output"
    print("    Overwriting the captured host buffer changes the result: the graph reads addresses")

    # Rebinding the context to a different device buffer does *not* change the graph, because the
    # graph recorded the kernel arguments at capture time and never consults the context again.
    _, other_address = cudart.cudaMalloc(worker.buffer["y"][1])
    cudart.cudaMemset(other_address, 0, worker.buffer["y"][1])
    worker.context.set_tensor_address("y", other_address)
    worker.run(1, True)
    rebound = np.empty(shape, dtype=np.float32)
    cudart.cudaMemcpy(rebound.ctypes.data, other_address, worker.buffer["y"][1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assert np.all(rebound == 0), "The graph must still write to the buffer it captured, not the new one"
    print("    Rebinding the context after capture leaves the graph untouched: it still writes to")
    print("    the old buffer, which is why a captured graph must own its buffers for its lifetime")

    cudart.cudaFree(other_address)
    worker.free()

if __name__ == "__main__":
    case_scaling()  # Throughput against thread count, with and without CUDA graph
    case_context_per_thread()  # One execution context per thread is mandatory, not an optimization
    case_graph_replays_addresses()  # A captured graph replays addresses, not bindings
    print("Finish")
