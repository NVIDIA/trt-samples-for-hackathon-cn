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
"""Record what the GPU was doing while a TensorRT engine ran.

Every timing example in the cookbook reports latency and nothing else. Latency alone cannot tell you
whether an engine is compute-bound, whether it fits in memory, or whether the GPU was busy with
somebody else's work while you measured.
"""

import time

import numpy as np
from gpu_telemetry import GpuTelemetry

from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

shape = [4, 1, 28, 28]
data = {"x": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}

def build_and_setup() -> TRTWrapperV1:
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw)
    assert tw.build(output_tensor_list), "Build failed"
    tw.setup(data, b_print_io=False)
    return tw

@case_mark
def case_telemetry_around_inference():
    """Sample while a loop of inferences runs, then report latency and telemetry together."""
    tw = build_and_setup()
    for _ in range(10):  # Warm up outside the measured region
        tw.infer(b_print_io=False)

    n_iteration = 300
    with GpuTelemetry(interval_ms=20) as telemetry:
        t0 = time.perf_counter()
        for _ in range(n_iteration):
            tw.infer(b_print_io=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"    {n_iteration} iterations in {elapsed_ms:.2f} ms -> {elapsed_ms / n_iteration:.4f} ms/iteration")
    print(telemetry.format_summary())
    assert telemetry.samples or telemetry.error is not None, "Either samples or a recorded error"

@case_mark
def case_build_is_where_the_memory_goes():
    """The same instrument on an engine build, which is the expensive half nobody measures.

    Inference on this network is a rounding error; the build is seconds of real GPU work. Pointing
    the sampler at the build shows a utilisation and memory profile that the inference loop does
    not, which is the argument for having the instrument at all.
    """
    with GpuTelemetry(interval_ms=50) as telemetry:
        t0 = time.perf_counter()
        tw = TRTWrapperV1()
        output_tensor_list = build_mnist_network_trt(tw)
        assert tw.build(output_tensor_list), "Build failed"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # `engine_bytes` is an IHostMemory, which has `nbytes` but no `len()`
    print(f"    Engine build took {elapsed_ms:.2f} ms, engine is {tw.engine_bytes.nbytes / (1 << 20):.2f} MiB")
    print(telemetry.format_summary())

@case_mark
def case_degrades_instead_of_failing():
    """The contract: telemetry that cannot run records an error and never raises.

    `mode="off"` takes the same path a container without `nvidia-smi` takes, which is the case this
    design exists for. The measured work must complete and the caller must still get its result.
    """
    with GpuTelemetry(mode="off") as telemetry:
        total = sum(i * i for i in range(100000))
    print(f"    Work completed under disabled telemetry, result={total}")
    print(f"    samples={len(telemetry.samples)}, error={telemetry.error!r}")
    assert telemetry.samples == [], "Disabled telemetry must collect nothing"
    assert telemetry.error is not None, "Disabled telemetry must say why it collected nothing"

    summary = telemetry.summary()
    assert summary["n_sample"] == 0 and summary["error"] is not None
    print(f"    summary() is still well formed: {summary}")

@case_mark
def case_sampling_cost_stays_outside():
    """Show that the sampler is beside the timed region, not in it.

    An `nvidia-smi` launch costs tens of milliseconds. If the loop below slowed down in proportion
    to the sampling rate, the instrument would be changing the number it reports.

    The loop has to be long enough that the two sampling rates actually produce different poll
    counts. At 300 iterations it finishes inside one 200 ms interval and both settings collect the
    same single poll, which compares nothing.
    """
    tw = build_and_setup()
    for _ in range(10):
        tw.infer(b_print_io=False)

    n_iteration = 20000
    result = {}
    for interval_ms in (20, 200):
        with GpuTelemetry(interval_ms=interval_ms) as telemetry:
            t0 = time.perf_counter()
            for _ in range(n_iteration):
                tw.infer(b_print_io=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary = telemetry.summary()
        result[interval_ms] = (elapsed_ms / n_iteration, summary.get("n_poll", 0))
        print(f"    interval={interval_ms:>4} ms -> {result[interval_ms][0]:.4f} ms/iteration, {result[interval_ms][1]} polls")

    fast, slow = result[20][0], result[200][0]
    ratio = max(fast, slow) / max(min(fast, slow), 1e-9)
    print(f"    10x the sampling rate changed per-iteration latency by {abs(ratio - 1) * 100:.1f}%")
    print("    Read this as a sanity check, not a guarantee: it is one unrepeated pair on a shared GPU.")

if __name__ == "__main__":
    case_telemetry_around_inference()
    case_build_is_where_the_memory_goes()
    case_degrades_instead_of_failing()
    case_sampling_cost_stays_outside()

    print("\nFinish")
