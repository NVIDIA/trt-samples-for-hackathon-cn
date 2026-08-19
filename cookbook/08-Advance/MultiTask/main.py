# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Serve several *different* engines at once: threads, streams, CUDA graphs and device pinning.

The cookbook already covers the ingredients separately -- `MultiContext` (several contexts of
one engine), `MultiStream` (one context, several streams), `MultiDevice` (one plan, several
GPUs), `CudaGraph` (replacing launch overhead with a graph replay). What none of them shows
is the thing an inference service actually is: **N different models running concurrently in
one process**, which is where those four features have to be combined and where they
interact.

Four configurations of the same workload, measured end to end:

1. `sequential`      - one thread, run each engine in turn. The baseline.
2. `thread`          - one thread and one stream per engine.
3. `thread_graph`    - same, plus each task captured into a CUDA graph.
4. `thread_graph_pinned` - same, with the tasks spread across GPUs.

The interesting questions are which of these actually pays, and what it costs to combine
them. Re-expressed from the idea in the internal `samples_internal/sampleMultiTasks`
(Apache-2.0); no code was taken from it.

Three things the code has to get right, each of which fails quietly:

+ **A CUDA graph capture must not be the first launch.** Capture records work; it does not
  run it. TensorRT allocates lazily on the first `execute_async_v3`, so capturing before a
  warm-up either fails outright or records the allocation. Warm up, then capture.
+ **`cudaSetDevice` is per-thread** -- a worker thread starts on device 0 regardless of what
  the parent did, so a "pinned" worker that forgets to set its device quietly runs everywhere.
+ **Capture must use a non-default stream.** `cudaStreamBeginCapture` on the legacy default
  stream is rejected, and the cookbook's wrappers default to stream 0.
"""

import threading
import time
from collections import OrderedDict

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_TASK = 4  # Number of distinct models served at once
N_WARMUP = 20
N_INFERENCE = 100

# Each task is a different model, deliberately: a service does not run four copies of one net
# Sized so that ONE task takes milliseconds, not microseconds. That matters: the threaded
# configurations pay a fixed orchestration cost (two Python barriers per round, ~0.4 ms for
# 4 threads), so on microsecond-scale engines the measurement is of Python, not of TensorRT.
# `case_orchestration_floor` measures that floor explicitly rather than hiding it.
TASK_SPEC = [
    {
        "name": "small",
        "n_channel": 128,
        "n_size": 256,
        "n_layer": 8
    },
    {
        "name": "medium",
        "n_channel": 256,
        "n_size": 256,
        "n_layer": 8
    },
    {
        "name": "wide",
        "n_channel": 384,
        "n_size": 128,
        "n_layer": 8
    },
    {
        "name": "deep",
        "n_channel": 128,
        "n_size": 256,
        "n_layer": 24
    },
]

result = OrderedDict()

# ================================================================ Helpers

def build_engine(spec: dict) -> bytes:
    """One convolution stack per task, with its own weights."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    n_channel, n_size = spec["n_channel"], spec["n_size"]
    tensor = network.add_input("input", trt.float32, [1, n_channel, n_size, n_size])
    for _ in range(spec["n_layer"]):
        weight = np.random.rand(n_channel, n_channel, 3, 3).astype(np.float32) * 0.02 - 0.01
        bias = np.zeros(n_channel, dtype=np.float32)
        layer = network.add_convolution_nd(tensor, n_channel, [3, 3], trt.Weights(np.ascontiguousarray(weight)), trt.Weights(bias))
        layer.padding_nd = [1, 1]
        tensor = network.add_activation(layer.get_output(0), trt.ActivationType.RELU).get_output(0)
    tensor.name = "output"
    network.mark_output(tensor)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    assert engine_bytes is not None, f"Failed building engine for {spec['name']}"
    return bytes(engine_bytes)

class Task:
    """One model, resident on one device, with its own stream, buffers and optional graph."""

    def __init__(self, spec: dict, engine_bytes: bytes, device: int) -> None:
        self.spec = spec
        self.device = device
        cudart.cudaSetDevice(device)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        # A non-default stream: capture is rejected on the legacy default stream
        self.stream = cudart.cudaStreamCreate()[1]
        self.graph_executable = None

        self.buffer = OrderedDict()
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = self.context.get_tensor_shape(name)
            n_byte = trt.volume(shape) * self.engine.get_tensor_dtype(name).itemsize
            address = cudart.cudaMalloc(n_byte)[1]
            self.buffer[name] = (address, n_byte)
            self.context.set_tensor_address(name, address)

        input_name = self.engine.get_tensor_name(0)
        host = np.random.rand(*self.context.get_tensor_shape(input_name)).astype(np.float32)
        cudart.cudaMemcpy(self.buffer[input_name][0], np.ascontiguousarray(host).ctypes.data, self.buffer[input_name][1], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    def run_once(self) -> None:
        """One inference, graph replay when a graph has been captured."""
        if self.graph_executable is None:
            self.context.execute_async_v3(self.stream)
        else:
            cudart.cudaGraphLaunch(self.graph_executable, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def capture_graph(self) -> None:
        """Warm up first, then capture. Capturing a cold context records the allocation."""
        cudart.cudaSetDevice(self.device)
        for _ in range(3):
            self.context.execute_async_v3(self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)
        self.context.execute_async_v3(self.stream)
        status, graph = cudart.cudaStreamEndCapture(self.stream)
        assert status == cudart.cudaError_t.cudaSuccess, f"Capture failed for {self.spec['name']}: {status}"
        status, self.graph_executable = cudart.cudaGraphInstantiate(graph, 0)[:2]
        assert status == cudart.cudaError_t.cudaSuccess, f"Instantiate failed for {self.spec['name']}: {status}"

    def free(self) -> None:
        cudart.cudaSetDevice(self.device)
        for address, _ in self.buffer.values():
            cudart.cudaFree(address)
        cudart.cudaStreamDestroy(self.stream)

def measure_sequential(task_list: list) -> float:
    """One thread, each task in turn. Wall time for one round over all tasks."""
    for _ in range(N_WARMUP):
        for task in task_list:
            cudart.cudaSetDevice(task.device)
            task.run_once()
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        for task in task_list:
            cudart.cudaSetDevice(task.device)
            task.run_once()
        latency_list.append((time.time() - t0) * 1000)
    return float(np.median(latency_list))

def measure_threaded(task_list: list) -> float:
    """One thread per task, all released together. Wall time until the slowest finishes."""
    barrier = threading.Barrier(len(task_list) + 1)
    done = threading.Barrier(len(task_list) + 1)
    stop = threading.Event()

    def worker(task: Task) -> None:
        cudart.cudaSetDevice(task.device)  # Per-thread; a worker starts on device 0 otherwise
        while True:
            barrier.wait()
            if stop.is_set():
                return
            task.run_once()
            done.wait()

    thread_list = [threading.Thread(target=worker, args=(task, ), daemon=True) for task in task_list]
    for thread in thread_list:
        thread.start()

    for _ in range(N_WARMUP):
        barrier.wait()
        done.wait()
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        barrier.wait()
        done.wait()
        latency_list.append((time.time() - t0) * 1000)

    stop.set()
    barrier.wait()
    for thread in thread_list:
        thread.join(timeout=5)
    return float(np.median(latency_list))

# ================================================================ Cases

@case_mark
def case_build() -> None:
    """Build the four engines once; every configuration reuses these bytes."""
    _, device_count = cudart.cudaGetDeviceCount()
    result["device_count"] = device_count
    cudart.cudaSetDevice(0)
    engine_bytes_list = []
    for spec in TASK_SPEC:
        engine_bytes = build_engine(spec)
        engine_bytes_list.append(engine_bytes)
        print(f"    {spec['name']:<8} channel={spec['n_channel']:<4} size={spec['n_size']:<3} layer={spec['n_layer']:<3} plan={len(engine_bytes) / 1024:.0f} KiB")
    result["engine_bytes_list"] = engine_bytes_list
    print(f"    GPUs visible: {device_count}")
    return

@case_mark
def case_orchestration_floor() -> None:
    """How much of any threaded number is Python rather than TensorRT.

    Runs the exact threading harness with a no-op instead of an inference. Whatever this
    costs is a floor under every threaded configuration below, and it is the reason
    concurrency does not pay for microsecond-scale engines.
    """

    class NullTask:
        device = 0

        def run_once(self) -> None:
            return

    # This measurement is itself noisy -- it depends on how the OS happens to schedule four
    # threads through two barriers -- so repeat it and report the spread instead of one number.
    sample_list = [measure_threaded([NullTask() for _ in range(N_TASK)]) for _ in range(5)]
    floor = float(np.median(sample_list))
    print(f"    {N_TASK} threads, two barriers per round, no GPU work: median {floor:.3f} ms "
          f"(5 repeats: min {min(sample_list):.3f}, max {max(sample_list):.3f})")
    print("    Any threaded configuration below is this plus the real work.")
    if max(sample_list) / max(min(sample_list), 1e-9) > 3.0:
        print("    NOTE: the spread is wide, so treat the floor-corrected column as indicative only.")
    result["floor"] = floor
    result["floor_range"] = (min(sample_list), max(sample_list))
    return

@case_mark
def case_sequential() -> None:
    """Baseline: everything on GPU 0, one after another."""
    task_list = [Task(spec, engine_bytes, 0) for spec, engine_bytes in zip(TASK_SPEC, result["engine_bytes_list"])]
    latency = measure_sequential(task_list)
    print(f"    {N_TASK} tasks, 1 GPU, 1 thread: {latency:.3f} ms per round")
    result["sequential"] = latency
    for task in task_list:
        task.free()
    return

@case_mark
def case_thread() -> None:
    """One thread and one stream per task, all on GPU 0."""
    task_list = [Task(spec, engine_bytes, 0) for spec, engine_bytes in zip(TASK_SPEC, result["engine_bytes_list"])]
    latency = measure_threaded(task_list)
    print(f"    {N_TASK} tasks, 1 GPU, {N_TASK} threads/streams: {latency:.3f} ms per round")
    result["thread"] = latency
    for task in task_list:
        task.free()
    return

@case_mark
def case_thread_graph() -> None:
    """Same, with each task's launch sequence captured into a CUDA graph."""
    task_list = [Task(spec, engine_bytes, 0) for spec, engine_bytes in zip(TASK_SPEC, result["engine_bytes_list"])]
    for task in task_list:
        task.capture_graph()
    latency = measure_threaded(task_list)
    print(f"    {N_TASK} tasks, 1 GPU, {N_TASK} threads + CUDA graph: {latency:.3f} ms per round")
    result["thread_graph"] = latency
    for task in task_list:
        task.free()
    return

@case_mark
def case_thread_graph_pinned() -> None:
    """Same again, with the tasks spread over the available GPUs."""
    device_count = result["device_count"]
    if device_count < 2:
        print(f"    Skip since no enough GPU is ready (need 2, get {device_count})")
        return
    device_list = [index % device_count for index in range(N_TASK)]
    task_list = [Task(spec, engine_bytes, device) for spec, engine_bytes, device in zip(TASK_SPEC, result["engine_bytes_list"], device_list)]
    for task in task_list:
        task.capture_graph()
    latency = measure_threaded(task_list)
    print(f"    {N_TASK} tasks pinned to GPUs {device_list} + CUDA graph: {latency:.3f} ms per round")
    result["thread_graph_pinned"] = latency
    for task in task_list:
        task.free()
    return

# ================================================================ Entrance

def main() -> None:
    case_build()
    case_orchestration_floor()
    case_sequential()
    case_thread()
    case_thread_graph()
    case_thread_graph_pinned()

    print("\n" + "=" * 96)
    print(f"{'Configuration':<38}{'ms / round':>13}{'vs seq':>10}{'minus floor':>14}{'vs seq':>10}")
    print("-" * 96)
    baseline = result["sequential"]
    floor = result.get("floor", 0.0)
    row_list = [
        ("sequential", "1 GPU, 1 thread (baseline)", 0.0),
        ("thread", f"1 GPU, {N_TASK} threads + streams", floor),
        ("thread_graph", f"1 GPU, {N_TASK} threads + CUDA graph", floor),
        ("thread_graph_pinned", f"{result['device_count']} GPUs, threads + graph", floor),
    ]
    for key, label, overhead in row_list:
        if key not in result:
            continue
        raw = result[key]
        corrected = max(raw - overhead, 1e-6)
        print(f"{label:<38}{raw:>13.3f}{baseline / raw:>9.2f}x{corrected:>14.3f}{baseline / corrected:>9.2f}x")
    print("-" * 96)
    print(f"Threading orchestration floor (no GPU work at all): {floor:.3f} ms per round.")
    print("The 'minus floor' column removes it, so the right-hand ratio is the GPU-side effect.")
    print("=" * 96)
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
