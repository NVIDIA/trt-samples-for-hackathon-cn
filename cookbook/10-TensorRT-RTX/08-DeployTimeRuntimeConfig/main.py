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

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
try:
    import tensorrt_rtx as trt
except ModuleNotFoundError:  # optional package, see this directory's README
    print("[SKIP] tensorrt_rtx is not installed (pip install tensorrt_rtx)")
    raise SystemExit(0)
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import case_mark

HERE = Path(__file__).parent
PLAN_FILE = HERE / "model.trt"
CACHE_FILE = HERE / "runtime.cache"

SHAPE_BUILD = (8, 3, 64, 64)  # the shape the profile is optimized for
SHAPE_OTHER = (13, 3, 64, 64)  # a shape the engine has never seen
N_CHANNEL = 64
N_LAYER = 6

STRATEGY = {
    "NONE": trt.DynamicShapesKernelSpecializationStrategy.NONE,
    "LAZY": trt.DynamicShapesKernelSpecializationStrategy.LAZY,
    "EAGER": trt.DynamicShapesKernelSpecializationStrategy.EAGER,
}

def build_plan() -> int:
    """Build the deploy artefact once. Nothing here is RTX-specific; the RTX part is the deploy side."""
    if PLAN_FILE.exists():
        return PLAN_FILE.stat().st_size
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    input_tensor = network.add_input("x", trt.float32, [-1, 3, 64, 64])
    profile = builder.create_optimization_profile()
    profile.set_shape("x", [1, 3, 64, 64], list(SHAPE_BUILD), [32, 3, 64, 64])
    builder_config.add_optimization_profile(profile)

    rng = np.random.default_rng(31193)
    tensor, n_input_channel, keep_alive = input_tensor, 3, []
    for _ in range(N_LAYER):
        kernel = np.ascontiguousarray(rng.normal(0, 0.05, (N_CHANNEL, n_input_channel, 3, 3)).astype(np.float32))
        bias = np.ascontiguousarray(rng.normal(0, 0.05, (N_CHANNEL, )).astype(np.float32))
        keep_alive += [kernel, bias]  # trt.Weights does not own the buffer
        convolution_layer = network.add_convolution_nd(tensor, N_CHANNEL, [3, 3], trt.Weights(kernel), trt.Weights(bias))
        convolution_layer.padding_nd = [1, 1]
        tensor = network.add_activation(convolution_layer.get_output(0), trt.ActivationType.RELU).get_output(0)
        n_input_channel = N_CHANNEL
    network.mark_output(tensor)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    if engine_bytes is None:
        raise RuntimeError("Fail building engine bytes")
    PLAN_FILE.write_bytes(bytes(engine_bytes))
    return PLAN_FILE.stat().st_size

def infer(engine, runtime_config, shape):
    """One inference from a cold execution context. Returns (wall seconds, output checksum)."""
    t0 = time.perf_counter()
    context = engine.create_execution_context(runtime_config)
    context.set_input_shape("x", list(shape))
    data = np.ascontiguousarray(np.random.default_rng(7).normal(0, 1, shape).astype(np.float32))
    output = np.empty(tuple(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
    _, device_input = cudart.cudaMalloc(data.nbytes)
    _, device_output = cudart.cudaMalloc(output.nbytes)
    cudart.cudaMemcpy(device_input, data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_tensor_address("x", int(device_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(device_output))
    _, stream = cudart.cudaStreamCreate()
    context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    elapsed = time.perf_counter() - t0
    cudart.cudaMemcpy(output.ctypes.data, device_output, output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaFree(device_input)
    cudart.cudaFree(device_output)
    cudart.cudaStreamDestroy(stream)
    return elapsed, float(output.sum())

def steady_state(engine, runtime_config, shape, n_iteration=200):
    """Median-of-nothing steady state: warm up, then time n_iteration enqueues on one context."""
    context = engine.create_execution_context(runtime_config)
    context.set_input_shape("x", list(shape))
    data = np.ascontiguousarray(np.random.default_rng(7).normal(0, 1, shape).astype(np.float32))
    output = np.empty(tuple(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
    _, device_input = cudart.cudaMalloc(data.nbytes)
    _, device_output = cudart.cudaMalloc(output.nbytes)
    cudart.cudaMemcpy(device_input, data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_tensor_address("x", int(device_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(device_output))
    _, stream = cudart.cudaStreamCreate()
    for _ in range(20):
        context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    t0 = time.perf_counter()
    for _ in range(n_iteration):
        context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    elapsed = (time.perf_counter() - t0) / n_iteration
    cudart.cudaMemcpy(output.ctypes.data, device_output, output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaFree(device_input)
    cudart.cudaFree(device_output)
    cudart.cudaStreamDestroy(stream)
    return elapsed, float(output.sum())

def sm_clock() -> str:
    """SM clock and temperature. Any latency here is meaningless without them, see 99-Todo §3.1."""
    process = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True)
    clock, temperature = (x.strip() for x in process.stdout.strip().split(","))
    return f"{clock} MHz / {temperature} C"

def child(mode: str, argument: str) -> None:
    """A fresh process. JIT state does not survive it, which is the entire point of the measurement."""
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(PLAN_FILE.read_bytes())
    runtime_config = engine.create_runtime_config()
    # The runtime cache must stay referenced from Python: dropping it after set_runtime_cache
    # segfaults the process, the config does not keep it alive.
    cache = runtime_config.create_runtime_cache()
    loaded = False
    if mode == "cache":
        if argument == "use" and CACHE_FILE.exists():
            loaded = bool(cache.deserialize(CACHE_FILE.read_bytes()))
        runtime_config.set_runtime_cache(cache)
        elapsed, checksum = infer(engine, runtime_config, SHAPE_BUILD)
        blob = runtime_config.get_runtime_cache().serialize()
        if argument == "fill":
            CACHE_FILE.write_bytes(bytes(blob))
        print(json.dumps({"loaded": loaded, "first": elapsed, "blob": int(blob.nbytes), "checksum": checksum}))
    elif mode == "strategy":
        runtime_config.dynamic_shapes_kernel_specialization_strategy = STRATEGY[argument]
        runtime_config.set_runtime_cache(cache)
        runs = [infer(engine, runtime_config, shape) for shape in (SHAPE_BUILD, SHAPE_OTHER, SHAPE_OTHER)]
        print(json.dumps({"runs": [{"n": s[0], "t": r[0], "checksum": r[1]} for s, r in zip((SHAPE_BUILD, SHAPE_OTHER, SHAPE_OTHER), runs)]}))
    return

def spawn(mode: str, argument: str, env: dict = None) -> dict:
    environment = dict(os.environ, **(env or {}))
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", mode, argument], capture_output=True, text=True, env=environment)
    if process.returncode != 0 or not process.stdout.strip():
        raise RuntimeError(f"child({mode},{argument}) failed rc={process.returncode}: {process.stderr[-400:]}")
    return json.loads(process.stdout.strip().splitlines()[-1])

@case_mark
def case_runtime_cache_across_processes():
    """The RTX runtime cache is a JIT-kernel cache. Its value only shows across process boundaries.

    There are two caches stacked here, and only one of them ships with your application:
      - the CUDA driver's own PTX->SASS cache, on disk in ~/.nv/ComputeCache, shared by every
        process on the machine and disabled by CUDA_CACHE_DISABLE=1;
      - TensorRT-RTX's IRuntimeCache, which you serialize and ship yourself.
    Measuring the second one without pinning the first is how you get an unreproducible number.
    """
    print(f"    clock now: {sm_clock()}")
    print("    driver cache   run                    loaded   first inference   cache blob")
    print("    " + "-" * 76)
    measured = {}
    for driver_label, env in [("enabled", {}), ("DISABLED", {"CUDA_CACHE_DISABLE": "1"})]:
        CACHE_FILE.unlink(missing_ok=True)
        rows = []
        for label, argument in [("cold, no cache file", "fill"), ("warm, cache file", "use")]:
            result = spawn("cache", argument, env)
            rows.append(result)
            print(f"    {driver_label:<14} {label:<21} {str(result['loaded']):<8} {result['first']*1e3:>10.1f} ms   {result['blob']:>10,}")
        assert rows[0]["loaded"] is False and rows[1]["loaded"] is True
        assert rows[1]["checksum"] == rows[0]["checksum"], "the runtime cache must not change the numerics"
        measured[driver_label] = rows
    print()
    for driver_label, rows in measured.items():
        print(f"    driver cache {driver_label:<9} -> runtime cache is worth {rows[0]['first']/rows[1]['first']:5.1f}x")
    isolated = measured["DISABLED"]
    print(f"\n    With the driver cache disabled the numbers are reproducible: a {isolated[0]['blob']:,} byte")
    print(f"    runtime-cache file turns a {isolated[0]['first']*1e3:.0f} ms first inference into {isolated[1]['first']*1e3:.0f} ms.")
    print("    With it enabled the cold number is whatever the shared on-disk cache happens to hold")
    print("    (155 to 654 ms across repeated runs here), so the ratio moves between ~7x and ~28x")
    print("    without anything about TensorRT changing. The warm number is stable either way --")
    print("    that is the one to quote. A server that restarts without shipping this file pays the")
    print("    compile again, and cannot rely on the driver cache to hide it.")
    return

@case_mark
def case_dynamic_shape_specialization():
    """NONE / LAZY / EAGER, each in a fresh process so no JIT state leaks between them."""
    print("    Each row is its own process, with an empty runtime cache.")
    print(f"    n={SHAPE_BUILD[0]} is the profile's opt shape, n={SHAPE_OTHER[0]} has never been seen.")
    print(f"\n    strategy   rep     n={SHAPE_BUILD[0]} first     n={SHAPE_OTHER[0]} first     n={SHAPE_OTHER[0]} again")
    print("    " + "-" * 66)
    summary, checksums = {}, set()
    for name in ["NONE", "LAZY", "EAGER"]:
        for rep in range(3):
            runs = spawn("strategy", name)["runs"]
            checksums.update(round(r["checksum"], 3) for r in runs)
            summary.setdefault(name, []).append([r["t"] for r in runs])
            print(f"    {name:<10} {rep}    " + "".join(f"{r['t']*1e3:>11.1f} ms" for r in runs))
        print()
    assert len(checksums) == 2, f"expected one checksum per shape, got {checksums}"
    new_shape = {k: np.median([r[1] for r in v]) for k, v in summary.items()}
    print(f"    median first inference at the unseen shape: " + ", ".join(f"{k} {v*1e3:.1f} ms" for k, v in new_shape.items()))
    assert new_shape["EAGER"] > new_shape["NONE"], "measurement changed, re-read the conclusion below"
    print(f"    EAGER is {new_shape['EAGER']/new_shape['NONE']:.1f}x *slower* than NONE at a shape the engine has")
    print("    not seen, which is the opposite of what the name suggests. Eager specialization")
    print("    compiles a shape-specialized kernel up front; on this engine that costs more than")
    print("    the generic kernel it replaces saves. All three agree numerically.")
    return

@case_mark
def case_cuda_graph_strategy():
    """WHOLE_GRAPH_CAPTURE removes launch overhead. That only helps if launch overhead is the cost."""
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(PLAN_FILE.read_bytes())
    print(f"    clock now: {sm_clock()}")
    print("    strategy               steady state   checksum")
    print("    " + "-" * 50)
    result = {}
    for name, value in [("DISABLED", trt.CudaGraphStrategy.DISABLED), ("WHOLE_GRAPH_CAPTURE", trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE)]:
        runtime_config = engine.create_runtime_config()
        runtime_config.cuda_graph_strategy = value
        elapsed, checksum = steady_state(engine, runtime_config, SHAPE_BUILD)
        result[name] = elapsed
        print(f"    {name:<22} {elapsed*1e3:>9.4f} ms   {checksum:.4f}")
    ratio = result["DISABLED"] / result["WHOLE_GRAPH_CAPTURE"]
    print(f"\n    ratio {ratio:.2f}x, clock now {sm_clock()}")
    print(f"    {N_LAYER} convolutions at {SHAPE_BUILD[0]}x{N_CHANNEL}x64x64 are compute bound, so there is no launch")
    print("    overhead left for graph capture to remove. Reach for it when the engine is many")
    print("    small kernels, not when it is a few large ones.")
    return

def tiny_plan(logger, *flags, compute_capability=None):
    """A 4x4 MatMul, the smallest thing that still exercises the builder."""
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    for flag in flags:
        builder_config.set_flag(flag)
    accepted = None
    if compute_capability is not None:
        # The slots must be allocated before they can be written. num_compute_capabilities is a
        # writable property, not a read-only count of what this build happens to support.
        builder_config.num_compute_capabilities = 1
        accepted = builder_config.set_compute_capability(compute_capability, 0)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    input_tensor = network.add_input("x", trt.float32, [1, 4])
    weight = np.ascontiguousarray(np.eye(4, dtype=np.float32))
    constant_layer = network.add_constant([4, 4], trt.Weights(weight))
    layer = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    network.mark_output(layer.get_output(0))
    return accepted, builder.build_serialized_network(network, builder_config)

@case_mark
def case_compute_capability_targeting():
    """Build here for a GPU that is somewhere else. The trap is how the slots get allocated."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder_config = trt.Builder(logger).create_builder_config()
    print(f"    ComputeCapability enumerators: {sorted(x for x in dir(trt.ComputeCapability) if x.startswith('SM'))}")
    print(f"    a fresh builder_config has num_compute_capabilities = {builder_config.num_compute_capabilities}")
    accepted = builder_config.set_compute_capability(trt.ComputeCapability.SM89, 0)
    print(f"    set_compute_capability(SM89, 0) with 0 slots -> {accepted}   <- a False return, not an exception")
    builder_config.num_compute_capabilities = 2  # writable: allocate the slots first
    accepted = builder_config.set_compute_capability(trt.ComputeCapability.SM89, 0)
    print(f"    after num_compute_capabilities = 2           -> {accepted}")

    print("\n    target        set_compute_capability   plan sha256[:16]")
    print("    " + "-" * 60)
    digest = {}
    for name in [None, "CURRENT", "SM75", "SM89", "SM120"]:
        capability = None if name is None else getattr(trt.ComputeCapability, name)
        ok, plan = tiny_plan(logger, compute_capability=capability)
        digest[str(name)] = hashlib.sha256(bytes(plan)).hexdigest()[:16]
        print(f"    {str(name):<13} {str(ok):<24} {digest[str(name)]}")
    assert len(set(digest.values())) == len(digest), "each target must produce a distinct plan"
    print(f"\n    All {len(digest)} plans differ, so the target really is baked in. Note the failure mode:")
    print("    at the default 0 slots set_compute_capability returns False, and the build then")
    print("    silently produces a plan for the current device instead of the one you asked for.")
    return

@case_mark
def case_unavailable_here():
    """What the RTX deploy story promises that this machine still cannot run."""
    logger = trt.Logger(trt.Logger.ERROR)
    device = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip()
    print(f"    device      : {device}")
    print(f"    tensorrt_rtx: {trt.__version__}")

    print("\n    a 4x4 MatMul, built with:")
    for label, flags, capability in [("no flags", (), None), ("REFIT", (trt.BuilderFlag.REFIT, ), None), ("STRIP_PLAN + REFIT", (trt.BuilderFlag.STRIP_PLAN, trt.BuilderFlag.REFIT), None), ("REFIT, target SM89", (trt.BuilderFlag.REFIT, ), trt.ComputeCapability.SM89), ("REFIT, target SM120", (trt.BuilderFlag.REFIT, ), trt.ComputeCapability.SM120)]:
        _, plan = tiny_plan(logger, *flags, compute_capability=capability)
        print(f"      {label:<22} -> {'None' if plan is None else f'{plan.nbytes:,} bytes'}")
    print("    (each failure logs 'Myelin ... CUDA error 222 loading a module' on stderr)")

    runtime = trt.Runtime(logger)
    print("\n    get_engine_validity, on plans that all deserialize and run:")
    for name in [None, "CURRENT", "SM89"]:
        capability = None if name is None else getattr(trt.ComputeCapability, name)
        _, plan = tiny_plan(logger, compute_capability=capability)
        validity = runtime.get_engine_validity(bytes(plan))
        engine = runtime.deserialize_cuda_engine(bytes(plan))
        print(f"      {str(name):<10} validity={str(validity):<40} deserialize={'ok' if engine else 'FAIL'}")
    print(f"    runtime.engine_header_size = {runtime.engine_header_size}")

    print("\n    So two things remain unavailable here. REFIT never builds, for any target, which")
    print("    takes the whole weightless-engine story (STRIP_PLAN + REFIT, ship, refit at deploy)")
    print("    with it. And get_engine_validity answers INVALID for every plan, including ones it")
    print("    then deserializes and runs -- a false negative, so it cannot be used as a preflight.")
    print("    TensorRT-RTX enumerates SM75/80/86/89/120/121 and this is an SM100 datacenter part,")
    print("    outside the set it targets. Re-run on an actual RTX GPU before trusting this list.")
    return

def main() -> None:
    print(f"plan: {build_plan():,} bytes")
    case_runtime_cache_across_processes()
    case_dynamic_shape_specialization()
    case_cuda_graph_strategy()
    case_compute_capability_targeting()
    case_unavailable_here()
    return

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
        print("\nFinish")
