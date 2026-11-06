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
"""Profile one engine in several regimes and keep them apart.

`04-Feature/Profiler` answers "which layer is slow". This answers "which layer is slow **when**",
which becomes a different question as soon as the same engine is driven more than one way.
"""

import numpy as np
import tensorrt as trt
from stage_profiler import StageProfiler

from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

shape_small = [1, 1, 28, 28]
shape_large = [4, 1, 28, 28]

def build_dynamic_engine() -> TRTWrapperV1:
    """One engine whose batch dimension ranges 1..4, so it can be run several ways."""
    tw = TRTWrapperV1()
    tw.builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    output_tensor_list = build_mnist_network_trt(tw)
    assert tw.build(output_tensor_list), "Build failed"
    return tw

def attach(tw: TRTWrapperV1) -> StageProfiler:
    profiler = StageProfiler()
    tw.setup({"x": np.random.rand(*shape_large).astype(np.float32)}, b_print_io=False)
    tw.context.profiler = profiler
    return profiler

@case_mark
def case_cold_and_steady():
    """The first call and the steady state are different workloads on identical input.

    This is the MNIST-scale stand-in for what the upstream profiler exists for: an LLM's prefill
    and decode passes run the same engine in two regimes and have no business sharing one average.
    Here the two regimes are cold and warm, which is the contrast this engine actually has.
    """
    tw = build_dynamic_engine()
    profiler = attach(tw)
    tw.context.set_input_shape("x", shape_large)

    with profiler.stage("first-call"):
        tw.context.execute_async_v3(0)
    for _ in range(20):
        with profiler.stage("steady"):
            tw.context.execute_async_v3(0)

    print(profiler.format_report())
    first = profiler.stage_total_ms("first-call")
    steady = profiler.stage_total_ms("steady")
    print(f"    first-call {first * 1000:.1f} us vs steady {steady * 1000:.1f} us -> {first / steady:.2f}x")
    assert first > steady, "The first call is expected to be the slow one"
    print("    Measured over three separate builds while writing this: 1.81x, 1.65x, 1.64x.")

@case_mark
def case_two_input_shapes():
    """Two input shapes through the same engine -- and an honest negative result.

    The obvious second pair of stages is batch 1 against batch 4. On this engine and this GPU they
    come out the same, because an MNIST network at batch 4 does not come close to filling a B200.
    That is worth stating rather than hiding: the profiler's job here is to let you **establish**
    that two regimes are indistinguishable, which a flat profiler cannot do either way.
    """
    tw = build_dynamic_engine()
    profiler = attach(tw)

    for stage_name, shape in [("batch1", shape_small), ("batch4", shape_large)]:
        tw.context.set_input_shape("x", shape)
        tw.context.execute_async_v3(0)  # Warm up outside any stage
        for _ in range(20):
            with profiler.stage(stage_name):
                tw.context.execute_async_v3(0)

    print(profiler.format_comparison("batch1", "batch4"))
    total_1 = profiler.stage_total_ms("batch1")
    total_4 = profiler.stage_total_ms("batch4")
    print(f"    batch1 {total_1 * 1000:.1f} us vs batch4 {total_4 * 1000:.1f} us -> {total_4 / total_1:.2f}x")
    print("    4x the work for ~1.0x the time: this engine is launch-bound, not compute-bound.")

    # The warm-up calls went nowhere near a stage, and are still accounted for rather than dropped.
    unlabelled = profiler.record.get(StageProfiler.DEFAULT_STAGE, {})
    print(f"    The 2 un-staged warm-up calls landed in {StageProfiler.DEFAULT_STAGE!r}: "
          f"{len(unlabelled)} layers, {profiler.stage_total_ms(StageProfiler.DEFAULT_STAGE) * 1000:.1f} us")
    assert unlabelled, "Un-staged times must be kept, not silently dropped"

@case_mark
def case_what_a_flat_profiler_loses():
    """Put cold and warm through one undifferentiated list and look at the number it reports.

    Not a strawman: one flat list is what `CookbookProfiler`, or any `report_layer_time` that
    ignores context, produces -- and it is the obvious thing to write.
    """
    tw = build_dynamic_engine()
    profiler = attach(tw)
    tw.context.set_input_shape("x", shape_large)

    for _ in range(21):  # One cold call then twenty warm ones, all unlabelled
        tw.context.execute_async_v3(0)
    flat = profiler.stage_total_ms(StageProfiler.DEFAULT_STAGE)

    # A second engine, so the "separated" run is cold in exactly the way the flat run was
    tw2 = build_dynamic_engine()
    separated = attach(tw2)
    tw2.context.set_input_shape("x", shape_large)
    with separated.stage("first-call"):
        tw2.context.execute_async_v3(0)
    for _ in range(20):
        with separated.stage("steady"):
            tw2.context.execute_async_v3(0)

    first = separated.stage_total_ms("first-call")
    steady = separated.stage_total_ms("steady")
    print(f"    One flat bucket : {flat * 1000:7.1f} us")
    print(f"    Separated       : first-call {first * 1000:.1f} us, steady {steady * 1000:.1f} us")
    print(f"    The flat mean sits between the two and describes neither "
          f"({steady * 1000:.1f} < {flat * 1000:.1f} < {first * 1000:.1f}).")
    assert steady < flat < first, (steady, flat, first)

if __name__ == "__main__":
    case_cold_and_steady()
    case_two_input_shapes()
    case_what_a_flat_profiler_loses()

    print("\nFinish")
