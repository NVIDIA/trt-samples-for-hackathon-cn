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
"""Version-compatible engines, the lean runtime, and what each one costs.

`CreateConfig(version_compatible=True)` produces a plan that a *newer* TensorRT
can deserialize, by stapling a copy of the lean runtime into the plan. That is
not a metaphor: `case_what_the_flag_puts_in_the_plan` shows the plan growing by
almost exactly the size of `libnvinfer_lean.so`, so a one-layer engine grows by
the same 105 MB that a real model does.

`exclude_lean_runtime=True` takes it back out, and then you owe the runtime at
load time -- that is what `LoadRuntime(path)` is for. One 105 MB library beside N
engines, instead of 105 MB inside each of them.

The cost lands at load time, not in the steady state:
`case_where_the_cost_actually_lands` measures 34x on deserialization and no
measurable difference in inference.

What cannot be shown here is the actual payoff, because that needs two TensorRT
versions and this image has one (11.1.0.106). Everything below is the mechanism
and the price, measured; the compatibility promise itself is taken on faith.
"""

import glob
import os
import time

import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, CreateNetwork, EngineBytesFromNetwork, EngineFromBytes, LoadRuntime, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
feed = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}
# The versioned SONAME is major.minor.patch, which is not what `trt.__version__`
# reports (it has a fourth component), so glob for the real file.
lean_library = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.*.*.*"))[-1]
N_WARMUP, N_TEST = 50, 500

def build(**config_kwargs) -> bytes:
    return bytes(EngineBytesFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig(**config_kwargs))())

@func.extend(CreateNetwork())
def tiny_network(builder, network) -> None:
    """As small as an engine gets: one elementwise add on 8 floats."""
    tensor = network.add_input("x", trt.float32, (8, ))
    layer = network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.SUM)
    layer.get_output(0).name = "y"
    network.mark_output(layer.get_output(0))

def build_tiny(**config_kwargs) -> bytes:
    return bytes(EngineBytesFromNetwork(tiny_network, config=CreateConfig(**config_kwargs))())

def latency(plan: bytes) -> float:
    with TrtRunner(EngineFromBytes(plan)) as runner:
        for _ in range(N_WARMUP):
            runner.infer(feed)
        start = time.perf_counter()
        for _ in range(N_TEST):
            runner.infer(feed)
    return (time.perf_counter() - start) * 1000 / N_TEST

def deserialize_ms(plan: bytes, runtime=None) -> float:
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        EngineFromBytes(plan, runtime=runtime)()
        best = min(best, (time.perf_counter() - start) * 1000)
    return best

@case_mark
def case_what_the_flag_puts_in_the_plan() -> None:
    """The plan grows by the size of the lean runtime, because that is what is in it.

    `libnvinfer_lean.so` is 104.845 MB on this image, and the difference between
    the two plans is 104.845 MB. There is no proportionality to model size at
    all: a 16 KB engine and a 13 MB engine both grow by the same 105 MB, so the
    smaller the model, the more absurd the ratio.
    """
    baseline = build()
    compatible = build(version_compatible=True)
    lean_size = os.path.getsize(lean_library)

    print(f"    baseline plan            : {len(baseline) / 1e6:8.3f} MB")
    print(f"    version_compatible plan  : {len(compatible) / 1e6:8.3f} MB  ({len(compatible) / len(baseline):.2f}x)")
    print(f"    difference               : {(len(compatible) - len(baseline)) / 1e6:8.3f} MB")
    print(f"    {os.path.basename(lean_library):<25}: {lean_size / 1e6:8.3f} MB  -- the plan carries a copy of it")
    print(f"    the two agree to within {abs((len(compatible) - len(baseline)) - lean_size)} bytes")
    tiny, tiny_compatible = build_tiny(), build_tiny(version_compatible=True)
    print(f"    a one-layer engine       : {len(tiny) / 1e6:8.3f} MB -> {len(tiny_compatible) / 1e6:8.3f} MB  ({len(tiny_compatible) / len(tiny):.0f}x)")
    print("    the surcharge is a flat ~105 MB, not a percentage -- the smaller the model, the worse the ratio")
    return

@case_mark
def case_getting_the_size_back() -> None:
    """`exclude_lean_runtime=True` returns the plan to baseline size.

    You then have to supply the runtime yourself at load time, which is the whole
    job of `LoadRuntime(path)`: it makes a bootstrap `trt.Runtime` and calls
    `load_runtime` on it. The trade is one shared 105 MB library against 105 MB
    per engine, so it wins as soon as you ship more than one plan.

    Setting `exclude_lean_runtime` without `version_compatible` is rejected by
    Polygraphy itself, with a sentence that says what to do -- worth contrasting
    with `../13-PerLayerPrecision/`, where the removed loaders fail from two
    layers down inside pybind.
    """
    baseline = build()
    compatible = build(version_compatible=True)
    excluded = build(version_compatible=True, exclude_lean_runtime=True)

    print(f"    baseline                       : {len(baseline) / 1e6:8.3f} MB")
    print(f"    version_compatible             : {len(compatible) / 1e6:8.3f} MB")
    print(f"    + exclude_lean_runtime         : {len(excluded) / 1e6:8.3f} MB  ({len(excluded) - len(baseline):+d} B vs baseline)")

    try:
        CreateConfig(exclude_lean_runtime=True)(trt.Builder(trt.Logger(trt.Logger.ERROR)), None)
    except Exception as e:
        print(f"    exclude_lean_runtime alone -> {type(e).__name__}: {str(e).splitlines()[0]}")

    runtime = LoadRuntime(lean_library)
    engine = EngineFromBytes(excluded, runtime=runtime)()
    print(f"    loaded the excluded plan with LoadRuntime({lean_library.split('/')[-1]}): I/O {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}")
    return

@case_mark
def case_where_the_cost_actually_lands() -> None:
    """Deserialization pays for the 105 MB; inference does not notice it.

    The plan is read and validated at load, so a 9x bigger plan is a ~40x slower
    load. Once the engine exists the kernels are the same kernels, and the three
    latencies are inside each other's noise.

    Which tells you where the flag hurts: process startup, container image size,
    and anything that loads engines on demand. Not throughput.
    """
    plans = {
        "baseline           ": build(),
        "version_compatible ": build(version_compatible=True),
        "+ exclude_lean     ": build(version_compatible=True, exclude_lean_runtime=True),
    }

    measured = {}
    for tag, plan in plans.items():
        measured[tag] = (len(plan) / 1e6, deserialize_ms(plan), latency(plan))
        print(f"    {tag}: {measured[tag][0]:7.2f} MB, deserialize {measured[tag][1]:7.2f} ms, infer {measured[tag][2]:.3f} ms")

    slow, fast = measured["version_compatible "][1], measured["baseline           "][1]
    print(f"    deserialization is {slow / fast:.1f}x slower with the runtime embedded")
    print(f"    inference differs by {abs(measured['version_compatible '][2] - measured['baseline           '][2]) * 1000:.0f} us -- noise")
    return

@case_mark
def case_deserializing_it_by_hand() -> None:
    """A plain `trt.Runtime` returns `None` for a version-compatible engine.

    Not an exception -- `deserialize_cuda_engine` hands back `None` and logs the
    reason, so code that does not check gets an `AttributeError` on the next line
    instead. The reason is that an embedded lean runtime is host code, and
    `IRuntime::getEngineHostCodeAllowed()` defaults to false.

    Polygraphy's `EngineFromBytes` sets `runtime.engine_host_code_allowed = True`
    for you, inside a bare `try/except AttributeError`, so this only bites you
    when you deserialize with raw TensorRT.

    The result inverts what the names suggest: the plan that *includes* the lean
    runtime is the one a plain runtime refuses, and the plan that *excludes* it
    loads fine -- because excluding it means falling back to the linked
    `libnvinfer`, which here is the same version that built the plan.
    """
    plans = {
        "baseline           ": build(),
        "version_compatible ": build(version_compatible=True),
        "+ exclude_lean     ": build(version_compatible=True, exclude_lean_runtime=True),
    }

    for tag, plan in plans.items():
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(plan)
        print(f"    raw trt.Runtime, host code not allowed, {tag}: {'engine' if engine else 'None'}")

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    runtime.engine_host_code_allowed = True
    engine = runtime.deserialize_cuda_engine(plans["version_compatible "])
    print(f"    same plan after engine_host_code_allowed = True                    : {'engine' if engine else 'None'}")
    print("    Polygraphy's EngineFromBytes always sets that flag, so it never sees this")
    return

@case_mark
def case_the_other_compatibility_axis() -> None:
    """Hardware compatibility is a separate flag with a separate, much smaller bill.

    `version_compatible` is about the TensorRT version; `hardware_compatibility_level`
    is about the GPU. `AMPERE_PLUS` gives up architecture-specific kernels so one
    plan runs on Ampere and later.

    On MNIST that costs 0.46 MB and nothing measurable in latency, which is an
    honest result rather than a reassuring one: this model's layers do not use
    the kernels the restriction takes away. A transformer with heavy tensor-core
    use is where the number would show up, and nothing here proves it would not.
    """
    for tag, kwargs in [
        ("baseline             ", {}),
        ("AMPERE_PLUS          ", dict(hardware_compatibility_level=trt.HardwareCompatibilityLevel.AMPERE_PLUS)),
        ("SAME_COMPUTE_CAPABILITY", dict(hardware_compatibility_level=trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY)),
    ]:
        plan = build(**kwargs)
        print(f"    {tag}: {len(plan) / 1e6:7.3f} MB, infer {latency(plan):.3f} ms")
    print("    the two flags compose: version + hardware compatibility is one plan for many versions and many GPUs")
    return

if __name__ == "__main__":
    case_what_the_flag_puts_in_the_plan()
    case_getting_the_size_back()
    case_where_the_cost_actually_lands()
    case_deserializing_it_by_hand()
    case_the_other_compatibility_axis()

    print("\nFinish")
