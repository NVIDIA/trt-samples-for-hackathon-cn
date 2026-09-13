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
"""How much GPU memory can you actually use, and in what order must you give it back?

Two questions that decide whether a deployment survives its first week, and that neither the
API nor `nvidia-smi` answers directly.

**1. `cudaMemGetInfo` reports free memory, not usable memory.** The difference is
fragmentation and driver reserve, and it is large enough to matter on embedded parts where
the margin is thin. The only reliable answer is empirical: allocate until it fails.

**2. Release order is not free.** A TensorRT object graph has real ownership edges --
context inside engine inside runtime, plus whatever CUDA context they were created under --
and destroying them out of order produces a crash *at exit*, far from the mistake. That is
the failure mode `08-Advance/GreenContext` documents for green contexts; this file shows the
plain version and what happens either way.

Both are re-expressed as **concepts only** from the internal
`testing/tools/{allocation_checker,trt_shutdown_test}/`, whose code is proprietary. Nothing
was copied.

This example allocates a lot of memory on purpose. It frees everything it takes, checks that
it did, and runs on the GPU with the most free memory to stay out of the way.
"""

import gc
import subprocess
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

CHUNK_MIB = 256  # Probe granularity; smaller finds a tighter ceiling but takes longer
SAFETY_MIB = 2048  # Stop this far short of exhausting the device, so the run stays polite
PROBE_CAP_MIB = 16384  # Do not try to take the whole GPU; 16 GiB is enough to see the gap
output_path = Path(__file__).parent

result = OrderedDict()

def memory_info(device: int) -> tuple:
    """`(free, total)` in MiB, as the driver reports it."""
    cudart.cudaSetDevice(device)
    _, free, total = cudart.cudaMemGetInfo()
    return free // (1 << 20), total // (1 << 20)

def pick_device() -> int:
    """The device with the most free memory, so this does not disturb other work."""
    _, count = cudart.cudaGetDeviceCount()
    return max(range(count), key=lambda index: memory_info(index)[0])

# ================================================================ Cases

@case_mark
def case_reported_vs_usable() -> None:
    """Allocate 256 MiB at a time until it fails, and compare with what was reported free.

    The gap is the answer to "can I trust `cudaMemGetInfo`". It is never zero, because the
    driver keeps a reserve and because a large allocation needs *contiguous* address space,
    which fragmentation can deny while plenty of total memory remains.
    """
    device = pick_device()
    free_mib, total_mib = memory_info(device)
    print(f"    GPU {device}: {free_mib} MiB free of {total_mib} MiB total, per cudaMemGetInfo")

    budget_mib = min(max(free_mib - SAFETY_MIB, CHUNK_MIB), PROBE_CAP_MIB)
    address_list = []
    allocated_mib = 0
    while allocated_mib + CHUNK_MIB <= budget_mib:
        status, address = cudart.cudaMalloc(CHUNK_MIB * (1 << 20))
        if status != cudart.cudaError_t.cudaSuccess:
            break
        address_list.append(address)
        allocated_mib += CHUNK_MIB

    free_after, _ = memory_info(device)
    print(f"    allocated {allocated_mib} MiB in {len(address_list)} chunks of {CHUNK_MIB} MiB "
          f"(stopped {SAFETY_MIB} MiB short on purpose)")
    print(f"    free now: {free_after} MiB -- reported free fell by {free_mib - free_after} MiB "
          f"for {allocated_mib} MiB requested")
    overhead = (free_mib - free_after) - allocated_mib
    print(f"    driver overhead on top of the request: {overhead} MiB")

    for address in address_list:
        cudart.cudaFree(address)
    free_restored, _ = memory_info(device)
    print(f"    after freeing everything: {free_restored} MiB free (started at {free_mib} MiB)")
    assert free_restored >= free_mib - CHUNK_MIB, "This example leaked device memory"
    result["device"] = device
    result["probe"] = (free_mib, allocated_mib, overhead)
    return

@case_mark
def case_fragmentation() -> None:
    """Free memory is not the same as one contiguous block of that size.

    Allocate many small blocks, free every other one, then ask for something the total free
    memory could satisfy but no single hole can. This is the state a long-running server
    reaches, and the reason a model that "fits" fails to load after a few hours.
    """
    device = result["device"]
    cudart.cudaSetDevice(device)
    small_mib = 64
    n_block = 64

    address_list = []
    for _ in range(n_block):
        status, address = cudart.cudaMalloc(small_mib * (1 << 20))
        if status != cudart.cudaError_t.cudaSuccess:
            break
        address_list.append(address)
    print(f"    allocated {len(address_list)} x {small_mib} MiB")

    for index in range(0, len(address_list), 2):  # free every other block
        cudart.cudaFree(address_list[index])
    freed_mib = small_mib * len(range(0, len(address_list), 2))
    free_now, _ = memory_info(device)
    print(f"    freed every other block: {freed_mib} MiB returned, {free_now} MiB reported free")

    # Ask for a single block the size of everything just freed
    status, big = cudart.cudaMalloc(freed_mib * (1 << 20))
    if status == cudart.cudaError_t.cudaSuccess:
        print(f"    a single {freed_mib} MiB allocation still succeeded -- the allocator coalesced the holes")
        cudart.cudaFree(big)
    else:
        print(f"    a single {freed_mib} MiB allocation FAILED although {free_now} MiB is free")
        print("    -> this is fragmentation: total free memory is not one usable block")

    for index in range(1, len(address_list), 2):
        cudart.cudaFree(address_list[index])
    return

@case_mark
def case_release_order() -> None:
    """Destroy a TensorRT object graph in the right order, and in the wrong one.

    Correct order is the reverse of creation: context, then engine, then runtime. The wrong
    order is not guaranteed to crash -- Python's reference counting hides much of it -- so
    the useful output is what each ordering *reports*, not whether it survives.
    """
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensor = network.add_input("input", trt.float32, [1, 16, 32, 32])
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))
    engine_bytes = builder.build_serialized_network(network, builder.create_builder_config())

    for order_name, order in [("reverse of creation (correct)", ["context", "engine", "runtime"]), ("runtime first (wrong)", ["runtime", "engine", "context"])]:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        holder = {"runtime": runtime, "engine": engine, "context": context}
        try:
            for key in order:
                del holder[key]
            gc.collect()
            print(f"    {order_name:<32} survived")
        except Exception as exception:  # noqa: BLE001
            print(f"    {order_name:<32} raised {type(exception).__name__}: {str(exception)[:60]}")
        holder.clear()
        gc.collect()

    print("    Python's reference counting keeps the objects alive until the last reference goes,")
    print("    so `del` in the wrong order is usually survivable here. In C++ it is not: see")
    print("    `tests/check_cpp_ownership.py` and the lifetime note in 08-Advance/GreenContext.")
    return

@case_mark
def case_exit_ordering() -> None:
    """The failure that only appears at process exit, demonstrated in a child.

    A TensorRT object that outlives the CUDA context it was created under cannot clean up.
    The destructor runs during interpreter shutdown, after the CUDA runtime has already gone.

    **The symptom is not stable.** Three identical runs of the child below produced:

        rc=0,   5 destructor errors logged
        rc=-11, 0 errors   (SIGSEGV, silent)
        rc=-11, 0 errors

    Either way the teardown is broken, but *which* evidence you get changes between runs --
    so a harness checking only the exit code misses the first case, and one checking only the
    log misses the other two. Run it in a child so both are observable.
    """
    script = textwrap.dedent("""
        import numpy as np, tensorrt as trt, cuda.bindings.runtime as cudart
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        t = network.add_input("input", trt.float32, [1, 16, 32, 32])
        L = network.add_activation(t, trt.ActivationType.RELU)
        L.get_output(0).name = "output"
        network.mark_output(L.get_output(0))
        plan = builder.build_serialized_network(network, builder.create_builder_config())
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        # Deliberately reset the device while TensorRT objects are still alive
        cudart.cudaDeviceReset()
        print("work finished, objects still alive at exit", flush=True)
    """)
    process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(output_path))
    combined = process.stdout + process.stderr
    destructor_error = [line for line in combined.splitlines() if "~Scoped" in line or "In ~" in line or "Cuda Runtime (In " in line]
    print(f"    child exit code: {process.returncode}" + ("" if process.returncode == 0 else "   <- killed by signal, after all its output"))
    print(f"    destructor errors printed: {len(destructor_error)}")
    for line in destructor_error[:2]:
        print(f"        {line.split('] ')[-1][:110]}")

    if destructor_error and process.returncode == 0:
        print("    -> errors logged, exit code 0. A harness that checks only the status sees nothing.")
    elif process.returncode != 0 and not destructor_error:
        print("    -> SIGSEGV with no message at all. A harness that checks only the log sees nothing.")

    # **Which of the two happens is not deterministic.** Measured over three identical runs:
    #     rc=0, 5 destructor errors | rc=-11 (SIGSEGV), 0 errors | rc=-11, 0 errors
    # So this cannot be asserted as one specific outcome -- only that the teardown is broken,
    # by one route or the other. Asserting the first outcome alone made this example fail
    # intermittently, which is exactly the kind of flaky test worth not writing.
    broken = bool(destructor_error) or process.returncode != 0
    assert broken, "Expected the early cudaDeviceReset to break TensorRT's teardown, one way or the other"
    result["exit_returncode"] = process.returncode
    result["destructor_error"] = len(destructor_error)
    return

def main() -> None:
    case_reported_vs_usable()
    case_fragmentation()
    case_release_order()
    case_exit_ordering()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
