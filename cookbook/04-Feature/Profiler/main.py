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

import time
from collections import OrderedDict

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import (CookbookProfiler, TRTWrapperV1, build_mnist_network_trt, case_mark, load_mnist_network_trt)

data = {"x": np.zeros([1, 1, 28, 28], dtype=np.float32)}

# `case_latency_report` runs the same MNIST network as `C++/main.cpp`, at the largest batch its
# optimization profile allows, so that the two implementations can be compared on one machine.
BATCH = 4
N_WARM_UP, N_ITERATION = 20, 200

@case_mark
def case_normal(b_emit_profile):
    tw = TRTWrapperV1()

    load_mnist_network_trt(tw)

    tw.build()
    tw.setup(data)

    my_profiler = CookbookProfiler()
    tw.context.profiler = my_profiler  # assign profiler to context

    # When `tw.context.enqueue_emits_profile` is True, all enqueue will be reported by Profiler.
    # Otherwise, only the ONE enqueue after call of `tw.context.report_to_profiler()` will be reported.
    tw.context.enqueue_emits_profile = b_emit_profile  # Default: True

    tw.infer(b_print_io=False)

    if not b_emit_profile:
        tw.context.report_to_profiler()  # We should enqueue once at least before this call

    tw.infer(b_print_io=False)

########################################################################################################################
# Latency report - the python counterpart of `C++/main.cpp`

class MedianProfiler(trt.IProfiler):
    """Collect one entry per layer per execution, then reduce with the median.

    `CookbookProfiler` above prints a line every time it is called, which is right for seeing the
    interface work. It is the wrong shape for measurement: `report_layer_time` fires once per layer
    per execution, so 200 executions of an 8-layer engine produce 1600 lines. Accumulating and
    reducing afterwards is what the upstream `LayerProfile::median` does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.record = OrderedDict()

    def report_layer_time(self, layer_name, time_ms) -> None:
        self.record.setdefault(layer_name, []).append(time_ms)

    def median_per_layer(self):
        """(name, median ms) sorted by descending median."""
        out = [(name, percentile_of(sorted(value), 50.0)) for name, value in self.record.items()]
        return sorted(out, key=lambda pair: pair[1], reverse=True)

def percentile_of(sorted_value, p):
    """Linear-interpolation percentile, the convention trtexec reports and `C++/main.cpp` mirrors.

    Deliberately not `np.percentile`: its default is also linear interpolation, but writing the two
    lines out keeps this identical to the C++ side by inspection rather than by trust.
    """
    if len(sorted_value) == 0:
        return 0.0
    rank = (p / 100.0) * (len(sorted_value) - 1)
    lower, upper = int(np.floor(rank)), int(np.ceil(rank))
    return sorted_value[lower] + (sorted_value[upper] - sorted_value[lower]) * (rank - lower)

def summarize(value):
    """min / max / mean / median / p90 / p95 / p99 and the coefficient of variation."""
    if len(value) == 0:
        return {}
    value = sorted(value)
    mean = float(np.mean(value))
    # Coefficient of variation: standard deviation as a percentage of the mean. This is the number
    # that says whether the run is worth quoting at all - a few percent means the measurement is
    # stable, while 30% is telling you about the machine rather than about the engine.
    stddev = float(np.std(value))  # Population standard deviation, matching the C++ side
    return {
        "min": value[0],
        "max": value[-1],
        "mean": mean,
        "median": percentile_of(value, 50.0),
        "p90": percentile_of(value, 90.0),
        "p95": percentile_of(value, 95.0),
        "p99": percentile_of(value, 99.0),
        "cv": stddev / mean * 100.0 if mean > 0 else 0.0,
    }

@case_mark
def case_latency_report():
    """Measure H2D / compute / D2H / enqueue separately, in-process, without trtexec.

    `07-Tool/trtexec/parse_export_json.py` already computes percentiles, but only *after* trtexec
    has written its JSON. An application that embeds TensorRT has no trtexec and no JSON; it has to
    measure itself, and that is what this case shows.
    """
    tw = TRTWrapperV1()
    tw.build(build_mnist_network_trt(tw))

    # Deserialize here rather than calling `tw.setup()`: this case manages its own pinned host
    # buffers, and the wrapper's pageable ones would make the H2D/D2H measurement meaningless
    runtime = trt.Runtime(tw.logger)
    engine = runtime.deserialize_cuda_engine(tw.engine_bytes)
    context = engine.create_execution_context()

    n_io = engine.num_io_tensors
    name_list = [engine.get_tensor_name(i) for i in range(n_io)]
    for name in name_list:  # The network has a dynamic batch, so a shape must be chosen first
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, [BATCH] + list(engine.get_tensor_shape(name))[1:])

    host_buffer, device_buffer, byte_list = [], [], []
    for name in name_list:
        # Not `sizeof(float)` for every tensor: the MNIST network's `z` output is int64
        n_byte = int(trt.volume(context.get_tensor_shape(name))) * engine.get_tensor_dtype(name).itemsize
        # Pinned host memory, or H2D/D2H measure the driver's staging copy rather than the transfer
        host_pointer = cudart.cudaHostAlloc(n_byte, cudart.cudaHostAllocDefault)[1]
        host_buffer.append(host_pointer)
        device_buffer.append(cudart.cudaMalloc(n_byte)[1])
        byte_list.append(n_byte)
        context.set_tensor_address(name, device_buffer[-1])

    stream = cudart.cudaStreamCreate()[1]
    event = {key: cudart.cudaEventCreate()[1] for key in ["h2d0", "h2d1", "compute0", "compute1", "d2h0", "d2h1"]}

    def run_once():
        """One inference, timed in four pieces. Returns (h2d, compute, d2h, enqueue) in ms."""
        cudart.cudaEventRecord(event["h2d0"], stream)
        for i, name in enumerate(name_list):
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpyAsync(device_buffer[i], host_buffer[i], byte_list[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaEventRecord(event["h2d1"], stream)

        cudart.cudaEventRecord(event["compute0"], stream)
        # The host cost of `execute_async_v3` itself, which CUDA events cannot see because they only
        # record positions in the stream. When this approaches `compute`, the engine is launch-bound
        # and `08-Advance/CudaGraph` is the answer, not a faster kernel.
        enqueue_start = time.perf_counter()
        context.execute_async_v3(stream)
        enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0
        cudart.cudaEventRecord(event["compute1"], stream)

        cudart.cudaEventRecord(event["d2h0"], stream)
        for i, name in enumerate(name_list):
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpyAsync(host_buffer[i], device_buffer[i], byte_list[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaEventRecord(event["d2h1"], stream)
        cudart.cudaStreamSynchronize(stream)

        return (
            cudart.cudaEventElapsedTime(event["h2d0"], event["h2d1"])[1],
            cudart.cudaEventElapsedTime(event["compute0"], event["compute1"])[1],
            cudart.cudaEventElapsedTime(event["d2h0"], event["d2h1"])[1],
            enqueue_ms,
        )

    for _ in range(N_WARM_UP):
        run_once()

    record = {key: [] for key in ["H2D", "compute", "D2H", "latency", "enqueue"]}
    for _ in range(N_ITERATION):
        h2d, compute, d2h, enqueue = run_once()
        record["H2D"].append(h2d)
        record["compute"].append(compute)
        record["D2H"].append(d2h)
        record["latency"].append(h2d + compute + d2h)
        record["enqueue"].append(enqueue)

    header = f"{'metric':>12} | {'min':>9}{'max':>10}{'mean':>10}{'median':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'cv':>9}"
    print(header)
    print("-" * len(header))
    summary = {}
    for key, value in record.items():
        summary[key] = summarize(value)
        r = summary[key]
        print(f"{key:>12} | {r['min']:9.4f}{r['max']:10.4f}{r['mean']:10.4f}{r['median']:10.4f}"
              f"{r['p90']:10.4f}{r['p95']:10.4f}{r['p99']:10.4f}{r['cv']:8.2f}%  ms")

    latency = summary["latency"]
    print(f"\nThroughput: {1000.0 / latency['median'] * BATCH:.4g} inference/s at the median")
    print(f"p99 / median = {latency['p99'] / latency['median']:.2f}x -- the tail the mean would have hidden")

    # ---- Per-layer profile, reduced with the median.
    # A second context: attaching a profiler serialises the layers so each can be timed, so the
    # numbers above and the numbers below must not come from the same measurement.
    context_profiled = engine.create_execution_context()
    for i, name in enumerate(name_list):
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context_profiled.set_input_shape(name, [BATCH] + list(engine.get_tensor_shape(name))[1:])
        context_profiled.set_tensor_address(name, device_buffer[i])
    profiler = MedianProfiler()
    context_profiled.profiler = profiler
    for _ in range(50):
        context_profiled.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    per_layer = profiler.median_per_layer()
    total = sum(value for _, value in per_layer)
    print(f"\n{'median ms':>10}{'share':>10}  layer")
    for name, value in per_layer:
        print(f"{value:10.4f}{value / total * 100:9.1f}%  {name}")
    print("-" * 100)
    print(f"{total:10.4f}           (sum of per-layer medians)")
    print("    The sum does not match the un-profiled compute time above, and that is expected:")
    print("    profiling serialises the layers. Per-layer numbers find the expensive layer; they")
    print("    are never a total to quote.")

    # `profiler = None` does not detach. Profiling is a one-way switch for the lifetime of an
    # `IExecutionContext`; profile in a context you are about to destroy, or keep a separate
    # un-profiled context for the hot path.
    #
    # The python binding fails *more quietly* than the C++ one here. In C++, `setProfiler(nullptr)`
    # returns and the error is visible; in python the assignment statement itself raises nothing at
    # all, so the only sign is the `Error Code 3` line the TensorRT logger emits. Reading
    # `context.profiler` back afterwards still returns the old object, and it keeps being called.
    n_call_before = sum(len(value) for value in profiler.record.values())
    context_profiled.profiler = None
    print(f"\n    After `context.profiler = None`, context.profiler is {type(context_profiled.profiler).__name__}, not None")
    context_profiled.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    n_call_after = sum(len(value) for value in profiler.record.values())
    print(f"    and it was still called {n_call_after - n_call_before} more times by the next execution.")

    for pointer in device_buffer:
        cudart.cudaFree(pointer)
    for pointer in host_buffer:
        cudart.cudaFreeHost(pointer)
    for value in event.values():
        cudart.cudaEventDestroy(value)
    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    case_normal(True)  # We can see the report two times
    case_normal(False)  # We can only see the report one time
    case_latency_report()

    print("Finish")
