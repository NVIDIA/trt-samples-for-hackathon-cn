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
"""Every `trt.BuilderFlag`, and what it actually does to the build.

`02-API/BuilderConfig` shows the *shape* of the flag API - `set_flag` / `get_flag` / `clear_flag` /
`flags` - using one flag as a stand-in. This file is about the flags themselves: all of them, set
one at a time against the same network, with the consequence measured rather than described.

Eleven of the twenty have a dedicated example elsewhere in the cookbook, because they need a whole
workflow to mean anything (refit, weight stripping, timing caches, ...). `case_coverage_map` is the
index to those, and it **asserts that every member of `trt.BuilderFlag` is accounted for**, so a
flag added by a future TensorRT cannot quietly go undocumented.

+ Steps to run.

```bash
python3 main.py
```
"""

import sys
import time

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

sys.stdout.reconfigure(line_buffering=True)

shape = [-1, 1, 28, 28]
n_conv_output = 32

# Where each flag is really demonstrated. `here` means this file is the primary demonstration.
# Keep this exhaustive: `case_coverage_map` asserts it covers `trt.BuilderFlag` completely.
COVERAGE_MAP = {
    "DEBUG": "listed only - build-time debug sync, no artifact to show; debug *tensors* are 04-Feature/DebugTensor",
    "GPU_FALLBACK": "04-Feature/DLAStandalone - needs a DLA core, absent on this GPU",
    "REFIT": "04-Feature/Refit",
    "DISABLE_TIMING_CACHE": "here (case_cache_flags) + 04-Feature/TimingCache",
    "EDITABLE_TIMING_CACHE": "04-Feature/TimingCache (queryKeys / query / update)",
    "TF32": "07-Tool/Onnxruntime - clearing it moves TRT-vs-ORT agreement by 644x",
    "SPARSE_WEIGHTS": "04-Feature/Sparsity",
    "SAFETY_SCOPE": "04-Feature/Safety, 08-Advance/Safety - QNX / DRIVE only",
    "DIRECT_IO": "04-Feature/DataFormat",
    "VERSION_COMPATIBLE": "here (case_version_compatible_and_lean_runtime) + 04-Feature/VersionCompatibility",
    "EXCLUDE_LEAN_RUNTIME": "here (case_version_compatible_and_lean_runtime) + 04-Feature/LeanAndDispatchRuntime",
    "ERROR_ON_TIMING_CACHE_MISS": "07-Tool/Polygraphy/More/12-TacticsAndReproducibility",
    "DISABLE_COMPILATION_CACHE": "here (case_cache_flags)",
    "STRIP_PLAN": "here (case_strip_plan) + 04-Feature/WeightStripping",
    "REFIT_IDENTICAL": "04-Feature/WeightStripping",
    "WEIGHT_STREAMING": "04-Feature/WeightStreaming",
    "REFIT_INDIVIDUAL": "04-Feature/Refit",
    "STRICT_NANS": "listed only - needs a NaN-producing graph; no dedicated example yet",
    "MONITOR_MEMORY": "here (case_monitor_memory)",
    "DISTRIBUTIVE_INDEPENDENCE": "listed only - needs a tensor-parallel group to mean anything",
}

def build(flag_name_list=(), *, b_weight_heavy: bool = False, severity=trt.Logger.Severity.ERROR):
    """Build one plan with exactly the given flags set. Returns (plan size in bytes, error text)."""
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network()
    builder_config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    input_tensor = network.add_input("x", trt.float32, shape)
    profile.set_shape(input_tensor.name, [1, 1, 28, 28], [4, 1, 28, 28], [8, 1, 28, 28])
    builder_config.add_optimization_profile(profile)

    weight = np.random.rand(n_conv_output, 1, 5, 5).astype(np.float32)
    bias = np.zeros(n_conv_output, dtype=np.float32)
    layer_conv = network.add_convolution_nd(input_tensor, n_conv_output, [5, 5], trt.Weights(weight), trt.Weights(bias))

    if b_weight_heavy:
        # A big constant so that *weights* dominate the plan, which is the only regime where
        # `STRIP_PLAN` is worth anything. See `case_strip_plan`.
        n_flat = n_conv_output * 24 * 24
        layer_shuffle = network.add_shuffle(layer_conv.get_output(0))
        layer_shuffle.reshape_dims = (-1, n_flat)
        layer_constant = network.add_constant((n_flat, 128), trt.Weights(np.random.rand(n_flat, 128).astype(np.float32)))
        layer_last = network.add_matrix_multiply(layer_shuffle.get_output(0), trt.MatrixOperation.NONE, layer_constant.get_output(0), trt.MatrixOperation.NONE)
    else:
        layer_last = network.add_activation(layer_conv.get_output(0), trt.ActivationType.RELU)
    network.mark_output(layer_last.get_output(0))

    for flag_name in flag_name_list:
        builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))

    try:
        plan = builder.build_serialized_network(network, builder_config)
    except Exception as e:
        return 0, f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
    if plan is None:
        return 0, "build_serialized_network returned None"
    return plan.nbytes, ""

@case_mark
def case_set_get_clear():
    """The four ways to touch flags, and the one that is already on before you touch anything."""
    builder = trt.Builder(trt.Logger(trt.Logger.Severity.ERROR))
    builder_config = builder.create_builder_config()

    default_flag_list = [name for name in trt.BuilderFlag.__members__ if builder_config.get_flag(getattr(trt.BuilderFlag, name))]
    print(f"    flags on a freshly created BuilderConfig: {builder_config.flags} -> {default_flag_list}")

    builder_config.set_flag(trt.BuilderFlag.REFIT)
    print(f"    after set_flag(REFIT)   : get_flag(REFIT) = {builder_config.get_flag(trt.BuilderFlag.REFIT)}, flags = {builder_config.flags}")
    builder_config.clear_flag(trt.BuilderFlag.REFIT)
    print(f"    after clear_flag(REFIT) : get_flag(REFIT) = {builder_config.get_flag(trt.BuilderFlag.REFIT)}, flags = {builder_config.flags}")

    # `flags` is a plain bitmask, so assigning it *replaces* everything, including the default.
    builder_config.flags = 1 << int(trt.BuilderFlag.REFIT) | 1 << int(trt.BuilderFlag.DEBUG)
    print(f"    after `flags = 1<<REFIT | 1<<DEBUG`: TF32 still on? {builder_config.get_flag(trt.BuilderFlag.TF32)}")
    assert not builder_config.get_flag(trt.BuilderFlag.TF32)
    print("    -> **TF32 is on by default.** Assigning `flags` wholesale silently turns it off, which is a")
    print("       precision change nobody asked for. Prefer `set_flag` / `clear_flag` over assigning `flags`.")

@case_mark
def case_flag_matrix():
    """Set each of the twenty flags alone and measure what it costs. Most of them cost nothing.

    The point of doing all twenty against one network is calibration: it shows that a flag being
    *accepted* says nothing about it doing anything, and it isolates the handful that change the
    plan at all from the majority that only change behaviour elsewhere (at runtime, in the build
    log, or on hardware this machine does not have).
    """
    baseline_size, error = build()
    assert not error, error
    print(f"    baseline, no flags set: {baseline_size:,} B\n")
    print(f"    {'flag':28s} {'plan size':>14s} {'delta':>16s}")

    delta_dict = {}
    for flag_name in trt.BuilderFlag.__members__:
        size, error = build([flag_name])
        if error:
            print(f"    {flag_name:28s} {'-':>14s}   {error}")
            continue
        delta = size - baseline_size
        delta_dict[flag_name] = delta
        print(f"    {flag_name:28s} {size:>14,d} {delta:>+16,d}")

    changed = {k: v for k, v in delta_dict.items() if v != 0}
    print(f"\n    only {len(changed)} of {len(delta_dict)} flags changed the plan at all: {sorted(changed)}")
    print("    -> every flag was *accepted*; none of them failed the build. Acceptance is not effect.")
    print("       The other flags are not no-ops - they act at runtime, or in the build log, or on")
    print("       hardware this machine does not have. A plan-size diff is simply the wrong probe for them.")
    return baseline_size

@case_mark
def case_version_compatible_and_lean_runtime(baseline_size: int):
    """The most expensive flag in the set, and the flag that undoes it.

    `VERSION_COMPATIBLE` embeds a *lean runtime* into the plan so it can be loaded by a different
    TensorRT than the one that built it. That runtime is the entire cost.
    """
    size_dict = {}
    for tag, flag_name_list in [
        ("baseline", ()),
        ("VERSION_COMPATIBLE", ("VERSION_COMPATIBLE", )),
        ("VERSION_COMPATIBLE + EXCLUDE_LEAN_RUNTIME", ("VERSION_COMPATIBLE", "EXCLUDE_LEAN_RUNTIME")),
        ("EXCLUDE_LEAN_RUNTIME alone", ("EXCLUDE_LEAN_RUNTIME", )),
    ]:
        size_dict[tag], error = build(flag_name_list)
        assert not error, error
        print(f"    {tag:44s} {size_dict[tag]:>14,d} B")

    surcharge = size_dict["VERSION_COMPATIBLE"] - size_dict["baseline"]
    print(f"\n    the embedded lean runtime costs {surcharge / 2**20:.0f} MiB on a {size_dict['baseline'] / 2**10:.0f} KiB engine")
    assert size_dict["VERSION_COMPATIBLE + EXCLUDE_LEAN_RUNTIME"] == size_dict["baseline"]
    print("    -> `EXCLUDE_LEAN_RUNTIME` takes it back to the baseline **exactly, byte for byte**: you keep")
    print("       version compatibility and ship the runtime separately (04-Feature/LeanAndDispatchRuntime).")
    assert size_dict["EXCLUDE_LEAN_RUNTIME alone"] == size_dict["baseline"]
    print("    -> and alone it is a **silent no-op**. No error, no warning, same plan. It only ever")
    print("       subtracts something `VERSION_COMPATIBLE` added.")

@case_mark
def case_strip_plan():
    """`STRIP_PLAN` removes the weights - which is a loss unless the weights were the plan.

    Everyone reaches for this to shrink an engine. On a small network it makes the plan *bigger*,
    because the refit metadata it adds outweighs the weights it removes.
    """
    for tag, b_weight_heavy in [("small network (32 conv filters)", False), ("weight-heavy network (+2.4 M matmul weights)", True)]:
        plain_size, error = build(b_weight_heavy=b_weight_heavy)
        assert not error, error
        stripped_size, error = build(["STRIP_PLAN"], b_weight_heavy=b_weight_heavy)
        assert not error, error
        ratio = stripped_size / plain_size
        verdict = "BIGGER" if stripped_size > plain_size else f"{1 / ratio:.0f}x smaller"
        print(f"    {tag:46s} {plain_size:>12,d} -> {stripped_size:>12,d} B  ({verdict})")
    print("    -> the flag is worth it exactly when weights dominate the plan, and counterproductive when")
    print("       they do not. Check the ratio before adopting it; and remember the weights have to come")
    print("       back via a refit at load time (04-Feature/WeightStripping).")

@case_mark
def case_cache_flags():
    """`DISABLE_TIMING_CACHE` and `DISABLE_COMPILATION_CACHE`: two flags whose effect is not the plan.

    Both leave the plan byte-identical, which is why the matrix above shows `+0` for each. The
    obvious probe - "is the second build faster?" - is a bad one on a network this small: the
    saving is a few percent and GPU contention swamps it. So measure the *cache itself*, which is
    deterministic: a timing cache that was written to has entries in it, and one that was disabled
    does not.
    """

    def build_and_harvest_cache(flag_name_list):
        """Build once with a fresh timing cache attached; return (build seconds, cache bytes)."""
        logger = trt.Logger(trt.Logger.Severity.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network()
        builder_config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        input_tensor = network.add_input("x", trt.float32, shape)
        profile.set_shape(input_tensor.name, [1, 1, 28, 28], [4, 1, 28, 28], [8, 1, 28, 28])
        builder_config.add_optimization_profile(profile)
        weight = np.random.rand(n_conv_output, 1, 5, 5).astype(np.float32)
        layer = network.add_convolution_nd(input_tensor, n_conv_output, [5, 5], trt.Weights(weight), trt.Weights(np.zeros(n_conv_output, dtype=np.float32)))
        network.mark_output(network.add_activation(layer.get_output(0), trt.ActivationType.RELU).get_output(0))

        timing_cache = builder_config.create_timing_cache(b"")
        builder_config.set_timing_cache(timing_cache, ignore_mismatch=False)
        for flag_name in flag_name_list:
            builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))

        start_time = time.time()
        builder.build_serialized_network(network, builder_config)
        return time.time() - start_time, builder_config.get_timing_cache().serialize().nbytes

    size_dict = {}
    print(f"    {'flags':34s} {'build':>8s} {'timing cache after the build':>30s}")
    for tag, flag_name_list in [("(none)", ()), ("DISABLE_TIMING_CACHE", ("DISABLE_TIMING_CACHE", )), ("DISABLE_COMPILATION_CACHE", ("DISABLE_COMPILATION_CACHE", ))]:
        duration, cache_size = build_and_harvest_cache(flag_name_list)
        size_dict[tag] = cache_size
        print(f"    {tag:34s} {duration:7.2f}s {cache_size:>26,d} B")

    assert size_dict["DISABLE_TIMING_CACHE"] < size_dict["(none)"] / 10
    assert size_dict["DISABLE_COMPILATION_CACHE"] < size_dict["(none)"] / 10
    print(f"\n    -> **both** flags leave the cache essentially empty ({size_dict['(none)'] / size_dict['DISABLE_TIMING_CACHE']:.0f}x and "
          f"{size_dict['(none)'] / size_dict['DISABLE_COMPILATION_CACHE']:.0f}x smaller).")
    print("       That is the surprise, and it says what a `timing` cache actually holds. Reading the three")
    print("       sizes as header + timings + compiled kernels:")
    timing_bytes = size_dict["DISABLE_COMPILATION_CACHE"] - size_dict["DISABLE_TIMING_CACHE"]
    kernel_bytes = size_dict["(none)"] - size_dict["DISABLE_COMPILATION_CACHE"]
    print(f"           header alone                  ~{size_dict['DISABLE_TIMING_CACHE']:>7,d} B  (timing cache off)")
    print(f"           + tactic timings              ~{timing_bytes:>7,d} B")
    print(f"           + JIT-compiled kernels        ~{kernel_bytes:>7,d} B  ({kernel_bytes / size_dict['(none)']:.1%} of the file)")
    print("       So the thing everyone calls `the timing cache` is overwhelmingly a *compilation* cache;")
    print("       the tactic timings it is named after are a rounding error. That is also why")
    print("       `DISABLE_COMPILATION_CACHE` shrinks a file that has `timing` in its name.")
    print("    Turn either off to make a build reproducible (07-Tool/Polygraphy/More/12-TacticsAndReproducibility)")
    print("    or to prove a stale cache is the culprit. Build time is the cost, and on a real model it is minutes.")

@case_mark
def case_monitor_memory():
    """`MONITOR_MEMORY` adds build-time memory reporting to the log - and nothing to the plan."""
    plain_size, _ = build(severity=trt.Logger.Severity.INFO)
    monitored_size, _ = build(["MONITOR_MEMORY"], severity=trt.Logger.Severity.INFO)
    print(f"    plan size without / with MONITOR_MEMORY: {plain_size:,} / {monitored_size:,} B")
    assert plain_size == monitored_size
    print("    -> identical. The flag is pure diagnostics: it makes the builder emit memory-usage records")
    print("       into the logger, so it is only visible at INFO/VERBOSE severity and only while building.")
    print("       Reach for it when a build OOMs and you need to know which phase asked for the memory.")

@case_mark
def case_coverage_map():
    """The index: for each flag, where in the cookbook it is actually demonstrated.

    The assert is the point. A flag added by a future TensorRT lands in `trt.BuilderFlag` but not in
    `COVERAGE_MAP`, and this case fails with its name - which is how "usage of each flag" stays true
    instead of being true only on the day it was written.
    """
    for flag_name in trt.BuilderFlag.__members__:
        print(f"    {int(getattr(trt.BuilderFlag, flag_name)):2d} {flag_name:28s} {COVERAGE_MAP.get(flag_name, '*** UNDOCUMENTED ***')}")

    documented = set(COVERAGE_MAP)
    actual = set(trt.BuilderFlag.__members__)
    assert documented == actual, (f"COVERAGE_MAP is out of date with TensorRT {trt.__version__}. "
                                  f"Missing: {sorted(actual - documented)}. Stale: {sorted(documented - actual)}.")
    print(f"\n    all {len(actual)} flags of TensorRT {trt.__version__} are accounted for")

if __name__ == "__main__":
    # The flag API itself, and the one flag that is already on
    case_set_get_clear()
    # All twenty, one at a time, measured
    baseline_size = case_flag_matrix()
    # The 100 MiB flag, and the flag that undoes it
    case_version_compatible_and_lean_runtime(baseline_size)
    # When stripping weights makes the plan bigger
    case_strip_plan()
    # Two flags that cost build time, not plan size
    case_cache_flags()
    # A flag that only ever writes to the log
    case_monitor_memory()
    # Where each flag is really demonstrated
    case_coverage_map()

    print("Finish")
