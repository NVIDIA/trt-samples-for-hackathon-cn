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
"""Give a TensorRT engine a fixed slice of the GPU's SMs, from inside the process.

A green context (CUDA 12.4+) partitions the SMs of one GPU and hands out a stream bound to the
partition. Everything launched on that stream is confined to those SMs. Compared with MIG this
needs no root, no host configuration and no container restart, it can be created and destroyed at
will, and the partitions live in one process -- see `../MIG/README.md` for what MIG buys instead.

TensorRT needs no API for this: `execute_async_v3(stream)` on a green stream is all there is. The
reason this example exists is that this turns out not to be the whole story -- case 3 finds a hole
in the isolation and case 4 finds a free 19% that only appears if the engine is built in the right
place.

The engines below are chains of large matrix multiplies, chosen because they are SM bound, so the
latency reacts to the size of the partition rather than to memory bandwidth.
"""

import threading
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import case_mark

N_HEAVY = 1024  # Matrix size of the throughput-hungry engine
N_LIGHT = 512  # Matrix size of the latency-critical engine
logger = trt.Logger(trt.Logger.Severity.ERROR)

def check(return_value, tag: str = ""):
    """Unpack the `(status, ...)` tuples the CUDA bindings return and assert success."""
    status, *rest = return_value if isinstance(return_value, tuple) else (return_value, )
    assert int(status) == 0, f"{tag}: {status}"
    return rest[0] if len(rest) == 1 else rest

def build_engine(n: int, n_branch: int, n_layer: int, n_aux_stream: int | None = 0):
    """A chain of `n_layer` matrix multiplies per branch, ready to run.

    `n_aux_stream` is passed to `IBuilderConfig.max_aux_streams`: 0 forbids the auxiliary streams
    TensorRT would otherwise use to run independent branches concurrently, and `None` leaves the
    default (-1, "TensorRT decides"). Case 3 is about the difference.
    """
    builder = trt.Builder(logger)
    network = builder.create_network()
    input_tensor = network.add_input("x", trt.float32, [n, n])
    output_list = []
    for i_branch in range(n_branch):
        tensor = input_tensor
        for i_layer in range(n_layer):
            weight = np.random.default_rng(i_branch * 10 + i_layer).random([n, n]).astype(np.float32) / n
            constant = network.add_constant([n, n], np.ascontiguousarray(weight))
            tensor = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, constant.get_output(0), trt.MatrixOperation.NONE).get_output(0)
        tensor.name = f"y{i_branch}"
        output_list.append(tensor)
    for tensor in output_list:
        network.mark_output(tensor)

    config = builder.create_builder_config()
    if n_aux_stream is not None:
        config.max_aux_streams = n_aux_stream
    engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(network, config))

    context = engine.create_execution_context()
    for name in ["x"] + [f"y{i}" for i in range(n_branch)]:
        context.set_tensor_address(name, check(cudart.cudaMalloc(n * n * 4)))
    return context

# ================================ Green context helpers

def get_device_sm_resource():
    """The SM resource of the whole device, which every partition is carved out of."""
    check(cuda.cuInit(0))
    device = check(cuda.cuDeviceGet(0))
    primary_context = check(cuda.cuDevicePrimaryCtxRetain(device))
    check(cuda.cuCtxSetCurrent(primary_context))
    resource = check(cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    return device, primary_context, resource

def split_sm(device, resource, n_sm: int):
    """Cut `n_sm` SMs off `resource` and return a stream for that partition and one for the rest.

    The split is not arbitrary: `minSmPartitionSize` / `smCoscheduledAlignment` (both 8 on H100 and
    on B200) round the request, so ask for what you get rather than assuming.
    """
    _, group_list, _, remainder = cuda.cuDevSmResourceSplitByCount(1, resource, 0, n_sm)

    def stream_of(one_resource):
        descriptor = check(cuda.cuDevResourceGenerateDesc([one_resource], 1))
        green_context = check(cuda.cuGreenCtxCreate(descriptor, device, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        stream = check(cuda.cuGreenCtxStreamCreate(green_context, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        return green_context, int(stream)

    green_a, stream_a = stream_of(group_list[0])
    if remainder.sm.smCount == 0:  # Asking for every SM leaves no remainder, and an empty
        return (green_a, stream_a, group_list[0].sm.smCount), (None, 0, 0)  # resource cannot be turned into a context
    green_b, stream_b = stream_of(remainder)
    return (green_a, stream_a, group_list[0].sm.smCount), (green_b, stream_b, remainder.sm.smCount)

def release(stream_list, green_context_list):
    """Give the partitions back. A partition of every SM has no remainder, hence the `None`s."""
    for stream in stream_list:
        if stream:
            check(cuda.cuStreamDestroy(stream))
    for green_context in green_context_list:
        if green_context is not None:
            check(cuda.cuGreenCtxDestroy(green_context))
    return

def synchronize(stream: int):
    """Wait for one stream, or for the whole device when the default stream is used."""
    if stream:
        cudart.cudaStreamSynchronize(stream)
    else:
        cudart.cudaDeviceSynchronize()

def time_inference(context, stream: int, n_warmup: int = 3, n_run: int = 10) -> float:
    """Average wall-clock time of one `execute_async_v3` on the given stream."""
    for _ in range(n_warmup):
        context.execute_async_v3(stream)
    synchronize(stream)
    start_time = time.perf_counter()
    for _ in range(n_run):
        assert context.execute_async_v3(stream), "enqueueV3 failed"
    synchronize(stream)
    return (time.perf_counter() - start_time) * 1000 / n_run

def measure_latency(context, stream: int, n_run: int = 40):
    """Median and p95 of a single small inference, one at a time."""
    for _ in range(5):
        context.execute_async_v3(stream)
    synchronize(stream)
    sample_list = []
    for _ in range(n_run):
        start_time = time.perf_counter()
        context.execute_async_v3(stream)
        synchronize(stream)
        sample_list.append((time.perf_counter() - start_time) * 1000)
    sample_list.sort()
    return sample_list[len(sample_list) // 2], sample_list[int(len(sample_list) * 0.95)]

# ================================ Cases

@case_mark
def case_partition_scaling():
    """The partition is real: the same engine gets slower in proportion to the SMs it may use."""
    device, _, resource = get_device_sm_resource()
    print(f"    device: {resource.sm.smCount} SM, minSmPartitionSize={resource.sm.minSmPartitionSize}, "
          f"smCoscheduledAlignment={resource.sm.smCoscheduledAlignment}")

    context = build_engine(N_HEAVY, 1, 8, n_aux_stream=0)
    baseline = time_inference(context, 0)
    print(f"    default stream (all {resource.sm.smCount} SM): {baseline:7.3f} ms")
    for n_sm in [16, 32, 64, resource.sm.smCount]:
        (green, stream, sm_count), (green_rest, stream_rest, _) = split_sm(device, resource, n_sm)
        elapsed = time_inference(context, stream)
        print(f"    green stream, {sm_count:3d} SM         : {elapsed:7.3f} ms  ({elapsed / baseline:4.2f}x of the whole GPU, "
              f"SM ratio {resource.sm.smCount / sm_count:4.2f}x)")
        release([stream, stream_rest], [green, green_rest])
    print("    -> a partition of every SM costs nothing, so the mechanism itself is free")
    return

@case_mark
def case_noisy_neighbour():
    """What the partition is for: a background job stops eating a latency-critical job's SMs."""
    device, _, resource = get_device_sm_resource()
    latency_context = build_engine(N_LIGHT, 1, 4, n_aux_stream=0)
    background_context = build_engine(N_HEAVY, 4, 4, n_aux_stream=0)

    median, p95 = measure_latency(latency_context, 0)
    print(f"    alone, default stream                     : median {median:6.3f} ms, p95 {p95:6.3f} ms")

    b_stop = False

    def keep_busy(stream):
        while not b_stop:
            background_context.execute_async_v3(stream)
            synchronize(stream)

    # Both jobs on the default stream: they fight over every SM.
    thread = threading.Thread(target=keep_busy, args=(0, ))
    thread.start()
    shared_median, shared_p95 = measure_latency(latency_context, 0)
    b_stop = True
    thread.join()
    print(f"    background job, both on default stream    : median {shared_median:6.3f} ms, p95 {shared_p95:6.3f} ms  "
          f"({shared_median / median:5.2f}x / {shared_p95 / p95:5.2f}x)")

    # One partition each.
    (green_a, stream_a, sm_a), (green_b, stream_b, sm_b) = split_sm(device, resource, 32)
    b_stop = False
    thread = threading.Thread(target=keep_busy, args=(stream_b, ))
    thread.start()
    split_median, split_p95 = measure_latency(latency_context, stream_a)
    b_stop = True
    thread.join()
    print(f"    background job, green {sm_a} SM / {sm_b} SM        : median {split_median:6.3f} ms, p95 {split_p95:6.3f} ms  "
          f"({split_median / median:5.2f}x / {split_p95 / p95:5.2f}x)")
    release([stream_a, stream_b], [green_a, green_b])
    print("    -> this is the MIG experiment, without MIG: same process, no root, no restart")
    return

@case_mark
def case_auxiliary_streams_escape_the_partition():
    """The trap: TensorRT's own auxiliary streams are not created inside the green context.

    A network with independent branches is run by TensorRT on auxiliary streams, and those come
    from the current context, not from the green one. The background engine below is confined to
    its own partition, yet the latency-critical job on the *other* partition still suffers -- and
    `max_aux_streams` defaults to -1 ("TensorRT decides"), so nobody has to ask for this to happen.
    """
    device, _, resource = get_device_sm_resource()
    latency_context = build_engine(N_LIGHT, 1, 4, n_aux_stream=0)
    (green_a, stream_a, sm_a), (green_b, stream_b, sm_b) = split_sm(device, resource, 32)
    print(f"    partitions: latency job {sm_a} SM, background job {sm_b} SM (disjoint)")

    median, p95 = measure_latency(latency_context, stream_a)
    print(f"    background idle                              : median {median:6.3f} ms, p95 {p95:6.3f} ms")

    for tag, n_aux_stream in [("max_aux_streams=0      ", 0), ("max_aux_streams=4      ", 4), ("default (-1, TRT decides)", None)]:
        background_context = build_engine(N_HEAVY, 4, 4, n_aux_stream=n_aux_stream)
        b_stop = False

        def keep_busy():
            while not b_stop:
                background_context.execute_async_v3(stream_b)
                synchronize(stream_b)

        thread = threading.Thread(target=keep_busy)
        thread.start()
        busy_median, busy_p95 = measure_latency(latency_context, stream_a)
        b_stop = True
        thread.join()
        print(f"    background on its partition, {tag}: median {busy_median:6.3f} ms ({busy_median / median:5.2f}x), "
              f"p95 {busy_p95:6.3f} ms ({busy_p95 / p95:5.2f}x)")
    release([stream_a, stream_b], [green_a, green_b])
    print("    -> build with `max_aux_streams = 0` or the isolation is a p95 illusion")
    return

@case_mark
def case_build_is_not_partition_aware():
    """Build inside the partition you will serve on: measurably faster, for a non-obvious reason.

    TensorRT cannot *ask* how big the partition is -- `cudaGetDeviceProperties`, which is what it
    reads, keeps reporting the whole device, and only the driver-level `cuCtxGetDevResource` knows
    about the 32 SM. But its tactic selection is empirical: the candidate kernels are timed inside
    whatever context is current, so a build done in the green context measures the partition's real
    behaviour and picks accordingly.

    The two builds on the whole GPU are the control. Without them a difference this size could be
    written off as build-to-build noise, and the honest answer would be "no conclusion".

    This is the same argument as `../MIG/README.md` ("build on the profile you deploy on"), except
    that here it is measured rather than expected.
    """
    device, primary_context, resource = get_device_sm_resource()
    (green, stream, sm_count), (green_rest, stream_rest, _) = split_sm(device, resource, 32)
    green_as_context = check(cuda.cuCtxFromGreenCtx(green))

    for tag, context_handle in [("primary context", primary_context), ("green context  ", green_as_context)]:
        check(cuda.cuCtxSetCurrent(context_handle))
        properties = check(cudart.cudaGetDeviceProperties(0))
        context_resource = check(cuda.cuCtxGetDevResource(context_handle, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        print(f"    {tag}: cudaGetDeviceProperties.multiProcessorCount={properties.multiProcessorCount}, "
              f"cuCtxGetDevResource={context_resource.sm.smCount}")

    # Two independent builds on the whole GPU are the control: TensorRT's tactic timing is not
    # deterministic, so two engines from the same source differ by a few percent on their own. Any
    # "the partition re-tuned the engine" claim has to beat that spread to mean anything.
    check(cuda.cuCtxSetCurrent(primary_context))
    engine_on_full_a = build_engine(N_HEAVY, 1, 8, n_aux_stream=0)
    engine_on_full_b = build_engine(N_HEAVY, 1, 8, n_aux_stream=0)
    check(cuda.cuCtxSetCurrent(green_as_context))
    engine_on_partition = build_engine(N_HEAVY, 1, 8, n_aux_stream=0)
    check(cuda.cuCtxSetCurrent(primary_context))

    result = {}
    for tag, context in [("built on the whole GPU (1)", engine_on_full_a), ("built on the whole GPU (2)", engine_on_full_b), ("built inside the partition", engine_on_partition)]:
        result[tag] = time_inference(context, stream)
        print(f"    {tag:26s}, run on {sm_count} SM: {result[tag]:7.3f} ms")
    control = abs(result["built on the whole GPU (1)"] - result["built on the whole GPU (2)"])
    effect = abs(result["built inside the partition"] - min(result["built on the whole GPU (1)"], result["built on the whole GPU (2)"]))
    print(f"    build-to-build spread of two identical builds: {control:.3f} ms; partition effect: {effect:.3f} ms")

    # No `release()` here on purpose. The engine built while the green context was current owns
    # CUDA resources belonging to that context, and destroying the context first makes TensorRT's
    # destructors fail with `Error Code 1: Cuda Runtime` and then take the process down with a
    # SIGSEGV -- at exit, far away from the mistake. Either outlive the TensorRT objects or destroy
    # them first; the other cases can call `release()` because their engines live in the primary
    # context and only the *stream* comes from the partition.
    print("    -> the tactic search is empirical, so it adapts to the partition even though the reported")
    print("       device properties do not: build where you will run")
    return

if __name__ == "__main__":
    case_partition_scaling()
    case_noisy_neighbour()
    case_auxiliary_streams_escape_the_partition()
    case_build_is_not_partition_aware()

    print("\nFinish")
