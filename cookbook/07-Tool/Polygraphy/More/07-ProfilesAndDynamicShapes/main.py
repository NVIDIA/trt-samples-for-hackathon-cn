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
"""Several optimization profiles in one engine, and whether choosing well matters.

A `Profile` maps each input to `min` / `opt` / `max` shapes. One engine can hold
several, and `TrtRunner(optimization_profile=N)` selects which one a runner uses.
The usual reason to build more than one is that a single `opt` cannot serve both
a latency-sensitive batch of 1 and an offline batch of 128.

Upstream builds three profiles and stops there. This example also **measures**
them, because "pick the profile that matches your shape" is only advice if the
difference is never shown.

The Torch-TensorRT side of the same topic is
`06-DLFrameworkTRT/Torch-TensorRT/DynamicShapes/`.
"""

import time

import numpy as np
from polygraphy.backend.trt import CreateConfig, Profile, TrtRunner, engine_from_network, network_from_onnx_path
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
N_WARMUP, N_TEST = 20, 100

# Three shapes of workload for the same model. `min == opt == max` pins a profile
# to one shape, which is what a latency-critical or an offline path usually wants.
profile_list = [
    Profile().add("x", min=(1, 1, 28, 28), opt=(1, 1, 28, 28), max=(1, 1, 28, 28)),
    Profile().add("x", min=(1, 1, 28, 28), opt=(4, 1, 28, 28), max=(32, 1, 28, 28)),
    Profile().add("x", min=(64, 1, 28, 28), opt=(64, 1, 28, 28), max=(64, 1, 28, 28)),
]
PROFILE_NAME = ["0: pinned to batch 1", "1: dynamic 1..32, opt 4", "2: pinned to batch 64"]

def data_of(batch: int) -> dict:
    """One input batch."""
    return {"x": np.random.rand(batch, 1, 28, 28).astype(np.float32)}

def benchmark(runner, feed: dict) -> float:
    """Mean latency in milliseconds."""
    for _ in range(N_WARMUP):
        runner.infer(feed)
    t0 = time.perf_counter()
    for _ in range(N_TEST):
        runner.infer(feed)
    return (time.perf_counter() - t0) * 1000 / N_TEST

engine = None

@case_mark
def case_one_engine_three_profiles() -> None:
    """Build once with all three profiles, and see what the engine reports."""
    global engine
    engine = engine_from_network(network_from_onnx_path(onnx_file), config=CreateConfig(profiles=profile_list))
    print(f"    engine holds {engine.num_optimization_profiles} optimization profile(s)")
    for index in range(engine.num_optimization_profiles):
        shape = engine.get_tensor_profile_shape("x", index)
        print(f"      profile {index} ({PROFILE_NAME[index].split(': ')[1]}): min={tuple(shape[0])} opt={tuple(shape[1])} max={tuple(shape[2])}")
    print("    one engine, three profiles -- the weights are stored once")
    return

@case_mark
def case_selecting_a_profile() -> None:
    """`TrtRunner(optimization_profile=N)`, and what each profile accepts.

    A profile is a hard range: a runner bound to profile 0 cannot be given
    batch 4, even though the engine as a whole supports it through profile 1.
    """
    for index in range(len(profile_list)):
        accepted = []
        for batch in [1, 4, 32, 64]:
            try:
                with TrtRunner(engine, optimization_profile=index) as runner:
                    runner.infer(data_of(batch))
                accepted.append(str(batch))
            except Exception:
                accepted.append(f"{batch}(x)")
        print(f"    {PROFILE_NAME[index]:<26}: batches {' '.join(accepted)}")
    print("    `(x)` means the shape is outside that profile's range, not outside the engine's")
    return

@case_mark
def case_does_the_choice_cost_anything() -> None:
    """Batch 1 through the pinned profile against the dynamic one.

    Both are valid, both give the same answer. The pinned profile was tuned for
    exactly this shape; the dynamic one was tuned for `opt=4` and has to cover
    1..32. Whether that costs anything is a property of the model -- measured
    rather than assumed.
    """
    feed = data_of(1)
    latency = {}
    for index in [0, 1]:
        with TrtRunner(engine, optimization_profile=index) as runner:
            latency[index] = benchmark(runner, feed)
        print(f"    batch 1 via {PROFILE_NAME[index]:<26}: {latency[index]:6.3f} ms")
    ratio = latency[1] / latency[0]
    print(f"    dynamic profile is {ratio:.2f}x the pinned one at its worst-case shape")
    print("    on this small model the tuning barely matters; on a large one it can")
    return

@case_mark
def case_same_numbers_from_every_profile() -> None:
    """Different profiles are different tactics, not different maths."""
    feed = data_of(1)
    reference = None
    for index in [0, 1]:
        with TrtRunner(engine, optimization_profile=index) as runner:
            output = np.array(runner.infer(feed)["y"])
        if reference is None:
            reference = output
            print(f"    profile 0 output taken as reference, shape {output.shape}")
        else:
            print(f"    profile {index} vs profile 0: max |diff| = {np.abs(output - reference).max():.3e}")
    return

if __name__ == "__main__":
    case_one_engine_three_profiles()
    case_selecting_a_profile()
    case_does_the_choice_cost_anything()
    case_same_numbers_from_every_profile()

    print("\nFinish")
