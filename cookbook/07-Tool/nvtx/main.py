# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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

from pathlib import Path

import numpy as np
import nvtx
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import (
    TRTWrapperV1,
    case_mark,
    load_mnist_network_trt,
)

trt_file = Path("model.trt")
data = {"x": np.arange(1 * 1 * 28 * 28, dtype=np.float32).reshape(1, 1, 28, 28)}
N_INFERENCE = 10

# A domain keeps the cookbook's events out of everyone else's rows in the
# timeline. Every API below takes `domain=` as a string, but see `case_domain`.
DOMAIN_NAME = "NVTX-cookbook"

tw = TRTWrapperV1()
load_mnist_network_trt(tw)
tw.build()
tw.setup(data)

def infer_once() -> None:
    """One inference, the thing every case below is annotating."""
    tw.context.execute_async_v3(0)
    return

@case_mark
def case_mark_and_range() -> None:
    """`mark` for an instant, and the three ways of marking a code range."""
    # A single instantaneous event
    nvtx.mark("build_done", color="black", domain=DOMAIN_NAME, category="setup")

    with nvtx.annotate("infer", color="yellow", domain=DOMAIN_NAME, category="multi-steps"):
        # 1. Context manager. Prefer this one: the range is closed even if the body raises.
        cudart.cudaDeviceSynchronize()
        for _ in range(N_INFERENCE):
            with nvtx.annotate("enqueue", color="red", domain=DOMAIN_NAME, category="step-red"):
                infer_once()
        cudart.cudaDeviceSynchronize()

        # 2. push / pop. Must be paired *in the same thread*, and a raise in between leaks the range.
        cudart.cudaDeviceSynchronize()
        for _ in range(N_INFERENCE):
            nvtx.push_range("enqueue", color="green", domain=DOMAIN_NAME, category="step-green")
            infer_once()
            nvtx.pop_range(domain=DOMAIN_NAME)
        cudart.cudaDeviceSynchronize()

        # 3. start / end. The only one that may cross threads, because the range is identified by the returned id rather than by a per-thread stack.
        cudart.cudaDeviceSynchronize()
        for _ in range(N_INFERENCE):
            range_id = nvtx.start_range("enqueue", color="blue", domain=DOMAIN_NAME, category="step-blue")
            infer_once()
            nvtx.end_range(range_id)
        cudart.cudaDeviceSynchronize()

    print(f"    {3 * N_INFERENCE} ranges in domain {DOMAIN_NAME}, one per inference")
    return

@case_mark
def case_decorator() -> None:
    """
    `annotate` also works as a decorator.
    The range is popped in a `finally`, so an exception cannot leak it.
    """

    @nvtx.annotate(color="orange", domain=DOMAIN_NAME, category="decorated")
    def preprocess(x):
        """`message` defaults to "preprocess", the function name."""
        return (x - x.mean()) / (x.std() + 1e-6)

    @nvtx.annotate("postprocess-renamed", color="purple", domain=DOMAIN_NAME, category="decorated")
    def postprocess(x):
        """An explicit message overrides the function name."""
        return x.argmax()

    for _ in range(N_INFERENCE):
        postprocess(preprocess(data["x"]))
    print("    the timeline shows `preprocess` (from the function name) and `postprocess-renamed`")
    return

@case_mark
def case_payload() -> None:
    """A `payload` carries the *data* of an event, separately from its message.

    Putting the batch size in the message (`f"enqueue-{n}"`) forces the tool to
    treat every distinct value as a different range name, which ruins grouping
    and statistics. A payload keeps one range name and one value per instance.
    """
    for i in range(N_INFERENCE):
        # int, float and bytes payloads need nothing extra
        with nvtx.annotate("enqueue", color="red", domain=DOMAIN_NAME, payload=i):
            infer_once()

    nvtx.mark("bytes_processed", domain=DOMAIN_NAME, payload=data["x"].nbytes)
    nvtx.mark("mean_activation", domain=DOMAIN_NAME, payload=float(data["x"].mean()))
    # Non-scalar payloads (list / tuple / range / bytes / numpy array) need NumPy
    nvtx.mark("input_shape", domain=DOMAIN_NAME, payload=np.array(data["x"].shape, dtype=np.int64))
    print(f"    {N_INFERENCE} `enqueue` ranges sharing one name, each carrying its own payload")
    return

@case_mark
def case_domain() -> None:
    """A `Domain` object plus a reused `EventAttributes`, the low-overhead path.

    Passing `domain="..."` to the module-level functions costs a dictionary
    lookup plus a fresh `EventAttributes` on *every* call. In a hot loop that
    overhead lands in the very measurement being taken. `nvtx.get_domain()`
    returns an object whose methods skip the lookup, and an `EventAttributes`
    built once can be reused, with only the fields that change being rewritten.
    """
    domain = nvtx.get_domain(DOMAIN_NAME)
    # Built once, outside the loop
    attributes = domain.get_event_attributes("enqueue-fast", color="cyan", category="low-overhead", payload=0)

    cudart.cudaDeviceSynchronize()
    for i in range(N_INFERENCE):
        domain.set_event_attributes(attributes, payload=i)  # Only the payload changes
        domain.push_range(attributes)
        infer_once()
        domain.pop_range()
    cudart.cudaDeviceSynchronize()

    # `mark` / `start_range` / `end_range` exist on the domain as well
    domain.mark(attributes, message="loop_done")
    range_id = domain.start_range(attributes, message="tail")
    domain.end_range(range_id)

    # Messages are registered strings: registered once, referenced by handle
    # afterwards, instead of being copied per event.
    print(f"    domain object: {type(domain).__name__} "
          f"(`DummyDomain` means no profiler is attached, so everything above was a no-op)")
    return

@case_mark
def case_counter() -> None:
    """Counters draw a *curve* on the timeline, not a range. Needs nvtx >= 0.2.16.

    Ranges answer "how long did this take". Counters answer "how did this value
    move over time" -- latency per iteration, throughput, occupancy, dynamic
    batch size. Nsight Systems plots them as their own rows.
    """
    domain = nvtx.get_domain(DOMAIN_NAME)

    # `get_counter` and `get_timestamp` arrived in nvtx 0.2.16 and are missing from *both* `Domain`
    # and `DummyDomain` before it, so an older nvtx fails here with a bare
    # `AttributeError: 'DummyDomain' object has no attribute 'get_counter'` -- which reads like the
    # "no profiler attached" case from `case_domain` above, but is really a version problem.
    # `requirements.txt` pins `nvtx>=0.2.16`; this keeps the diagnosis one line away.
    if not hasattr(domain, "get_counter"):
        print(f"    nvtx is too old for the counter API (need >= 0.2.16), skip. Run `pip install -U nvtx`.")
        return

    # `int` -> Int64Counter, `float` -> Float64Counter, a NumPy dtype -> ExtCounter
    iteration_counter = domain.get_counter("iteration", int, description="inference index")
    # `semantics` tells the tool the unit and the range to plot against
    latency_counter = domain.get_counter(
        "latency",
        float,
        description="wall time of one execute_async_v3",
        semantics=nvtx.CounterSemantics(unit="ms", min=0.0),
    )
    # A structured dtype records several fields as one grouped sample
    io_dtype = np.dtype([("input_byte", np.int64), ("output_byte", np.int64)])
    io_counter = domain.get_counter("io_byte", io_dtype, description="bytes moved per inference")

    cudart.cudaDeviceSynchronize()
    for i in range(N_INFERENCE):
        t0 = domain.get_timestamp()  # Tool provided timestamp, no clock of our own
        infer_once()
        cudart.cudaDeviceSynchronize()
        t1 = domain.get_timestamp()

        iteration_counter.sample(i)
        latency_counter.sample((t1 - t0) / 1e6)  # ns -> ms
        io_counter.sample(np.array((data["x"].nbytes, 10 * 4), dtype=io_dtype))

    # Recording "there is no value right now" is different from recording a 0
    iteration_counter.sample_no_value(nvtx.CounterNoValueReason.UNCHANGED)

    print(f"    counters: {type(iteration_counter).__name__} / {type(latency_counter).__name__}"
          f" / {type(io_counter).__name__} (`Dummy*` means no profiler is attached)")
    print("    `nsys stats --report nvtx_sum` does not show counters, they land in the")
    print("    `GENERIC_EVENTS` table of the report and as plot rows in the Nsight GUI.")
    return

@case_mark
def case_auto_profile() -> None:
    """`nvtx.Profile` annotates every Python call by itself, no source changes.

    Useful as a first pass, when it is not yet clear where the time goes. It uses
    `sys.setprofile`, so it is *slow* -- a diagnostic, not something to leave on.
    The whole script can be run this way too:

        nsys profile -o py python3 -m nvtx main.py

    Ranges land in a domain called `nvtx.py`, named `file.py:lineno(function)`.
    """
    profile = nvtx.Profile(linenos=True, annotate_cfuncs=False)
    profile.enable()
    for _ in range(N_INFERENCE):
        infer_once()
    profile.disable()
    print("    every Python call in the loop above got its own range in domain `nvtx.py`")
    return

@case_mark
def case_switch() -> None:
    """`NVTX_DISABLE=1` turns the whole library into no-ops at import time."""
    print(f"    nvtx.enabled() = {nvtx.enabled()}")
    print("    Set `NVTX_DISABLE=1` before starting python to strip every annotation:")
    print("    the module-level functions become empty bodies and `annotate` becomes")
    print("    `contextlib.nullcontext`, so no branch is needed at the call sites.")
    return

if __name__ == "__main__":
    case_mark_and_range()
    case_decorator()
    case_payload()
    case_domain()
    case_counter()
    case_auto_profile()
    case_switch()

    print("\nFinish")
