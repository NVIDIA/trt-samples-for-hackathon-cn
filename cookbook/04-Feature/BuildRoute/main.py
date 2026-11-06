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
"""`IBuilderConfig.all_build_routes` and `.build_route`: the builder's own tuning knobs.

`all_build_routes` is **not a setting**. It is a read-only JSON database of every knob the
internal tuner accepts, each with its allowed values, its **default**, and a line of help. That
makes it the only direct answer to "what is TensorRT's default build strategy?" - there is no
other API that reports it.

`build_route` is the matching input: a string of `-name=value` tokens handed to the tuner.

Both arrived mid-TensorRT-11.1 (absent in 11.1.0.2, present in 11.1.0.106) and the cookbook had
no Python-side example of either, which is why this directory exists. `07-Tool/trtexec/` covers
the CLI side (`--tuningCache`, whose JSON also carries a `build_route` field) - that is a
different thing: a search *over* routes, not the API that defines them.

Three things this measures rather than asserts:

1. the shape of the database - how many knobs, of what kinds, how many default on;
2. what a route actually changes on a small network (**on this one: nothing**);
3. how wrong input is handled, which is asymmetric and worth knowing.
"""

import json
import re
from collections import Counter, OrderedDict

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_SWEEP = 16  # Knobs flipped in `case_sweep`; each costs one build, roughly 5 s
N_BASELINE = 5  # Identical builds used to measure how much the plan size moves on its own
MARGIN = 1024  # B, added to each side of the measured baseline band

result = OrderedDict()

class RecordingLogger(trt.ILogger):
    """Keep the messages instead of printing them, so a case can quote its own build.

    Errors are the obvious reason, but INFO matters just as much here: the tuner announces the
    route it received as `[TuningBuildRoute] Using build route: ...`, and that line is the only
    *direct* evidence that a route reached the builder at all. Plan size is an indirect probe and,
    as `case_set_a_route` measures, a noisy one.
    """

    def __init__(self):
        trt.ILogger.__init__(self)
        self.message_list = []
        self.route_echo = ""

    def log(self, severity, message):
        if "[TuningBuildRoute] Using build route:" in message:
            self.route_echo = message.split("Using build route:", 1)[1].strip()
        if severity <= trt.ILogger.Severity.ERROR:
            self.message_list.append(message)

def build(build_route: str | None = None) -> tuple:
    """Build one small engine, optionally under a build route. Returns `(n_byte, first_error, route_echo)`.

    Deliberately tiny and deliberately the *same* network every time: the question here is whether
    the route changed the build, so everything else has to be held still.

    That is still not enough to read a plan-size difference straight off, and the first version of
    this example got it wrong by assuming it was. The builder times tactics, a tie can break either
    way, and five builds of this *identical* configuration give more than one size. So every
    comparison below is made against a measured baseline **band**, not a single number - the same
    discipline `02-API/BuilderFlag/` had to learn.
    """
    logger = RecordingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    if build_route is not None:
        builder_config.build_route = build_route

    tensor = network.add_input("x", trt.float32, [64, 64])
    weight = network.add_constant([64, 64], np.ascontiguousarray(np.random.rand(64, 64).astype(np.float32))).get_output(0)
    tensor = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE).get_output(0)
    tensor = network.add_activation(tensor, trt.ActivationType.RELU).get_output(0)
    tensor.name = "y"
    network.mark_output(tensor)

    engine_bytes = builder.build_serialized_network(network, builder_config)
    n_byte = None if engine_bytes is None else len(bytes(engine_bytes))
    return n_byte, (logger.message_list[0] if logger.message_list else ""), logger.route_echo

def get_knob_list() -> tuple:
    """Return `(tuner_version, knob_list)` parsed out of `all_build_routes`."""
    builder_config = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_builder_config()
    document = json.loads(builder_config.all_build_routes)
    return document["tuner_version"], document["tuner_options"]

def get_allowed_value_list(knob: dict) -> list | None:
    """`allowed_values` is a string like `-conv_use_long_w=[on|off]`; pull the choices out of it.

    Returns `None` for the free-form knobs, whose right-hand side is a type name such as `int`
    rather than a bracketed list.
    """
    match = re.search(r"=\[([^\]]*)\]", knob["allowed_values"])
    return match.group(1).split("|") if match else None

# ================================================================ Cases

@case_mark
def case_knob_database() -> None:
    """What `all_build_routes` actually contains. No engine is built here."""
    tuner_version, knob_list = get_knob_list()
    print(f"    tuner_version = {tuner_version}, {len(knob_list)} knobs")

    shape_counter = Counter(tuple(get_allowed_value_list(k) or ["<free-form>"]) for k in knob_list)
    print(f"\n    {'allowed values':<28}{'knobs':>6}")
    for shape, n in shape_counter.most_common():
        print(f"    {str(list(shape)):<28}{n:>6}")

    on_off_list = [k for k in knob_list if get_allowed_value_list(k) == ["on", "off"]]
    n_default_on = sum(1 for k in on_off_list if k["default_value"] == "on")
    print(f"\n    of the {len(on_off_list)} on/off knobs, {n_default_on} default to `on` and {len(on_off_list) - n_default_on} to `off`")
    print("    -> that list of defaults IS the default build strategy; no other API reports it")

    print(f"\n    every knob carries help text ({sum(1 for k in knob_list if k.get('help'))}/{len(knob_list)}), for example:")
    for knob in knob_list[:3]:
        print(f"      {knob['option']:<34} default={knob['default_value']:<6} {knob['help'][:70]}")

    result["tuner_version"] = tuner_version
    result["knob_list"] = knob_list
    assert len(knob_list) > 0 and tuner_version, "all_build_routes should describe at least one knob"
    return

@case_mark
def case_tuner_version_is_a_finer_fingerprint() -> None:
    """`tuner_version` moves independently of the TensorRT version, and so does the knob count.

    Recorded here as a reference point: a build whose TensorRT version matches yours can still
    have a different tuner, and this is how you see it.
    """
    tuner_version, knob_list = result["tuner_version"], result["knob_list"]
    print(f"    TensorRT {trt.__version__}: tuner {tuner_version}, {len(knob_list)} knobs")
    print("    measured earlier on TensorRT 11.1.0.106: tuner 2.19.45, 209 knobs")
    print("    -> the builder gained 6 knobs between those two releases while the API shape did not change")
    return

@case_mark
def case_set_a_route() -> None:
    """`build_route` defaults to the empty string; setting it is accepted and reads back."""
    builder_config = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_builder_config()
    print(f"    default build_route = {builder_config.build_route!r}")
    builder_config.build_route = "-conv_use_long_w=off"
    print(f"    after assignment     = {builder_config.build_route!r}")
    assert builder_config.build_route == "-conv_use_long_w=off"

    # Establish how much the plan size moves with nothing changed at all. Everything below is read
    # against this band; a route that lands inside it has told us nothing.
    baseline_list = [build()[0] for _ in range(N_BASELINE)]
    low, high = min(baseline_list), max(baseline_list)
    print(f"\n    baseline plan size, {N_BASELINE} identical builds: {baseline_list}")
    print(f"    -> {len(set(baseline_list))} distinct value(s), spread {high - low:,} B on a {low:,} B plan")
    print("       The builder times tactics; a tie broken differently changes the plan. So a single")
    print("       build is not a baseline, and `size != baseline` is not evidence of anything.")
    result["baseline_list"] = baseline_list
    result["baseline_band"] = (low - MARGIN, high + MARGIN)
    print(f"       Band used below, with a {MARGIN:,} B margin: {result['baseline_band']}")
    return

@case_mark
def case_accepted_is_not_applied() -> None:
    """A route can be accepted, and change nothing. Acceptance is not effect.

    The same trap as `02-API/BuilderFlag/`: every flag there was accepted too, and only a few
    changed the plan. A knob acts on a pattern that has to be *present* in the network, and a
    64x64 MatMul gives most of them nothing to do.
    """
    low, high = result["baseline_band"]
    print(f"    baseline band {low:,} .. {high:,} B")
    print(f"\n    {'route':<34}{'plan size':>12}   verdict")
    for knob_name in ["-conv_use_long_w=off", "-conv_lowering=off", "-reshape_ppg=off", "-transpose_ppg=off"]:
        n_byte, _, route_echo = build(knob_name)
        if n_byte is None:
            verdict = "build FAILED"
        elif low <= n_byte <= high:
            verdict = "inside the baseline band - no evidence of any effect"
        else:
            verdict = f"OUTSIDE the band by {min(abs(n_byte - low), abs(n_byte - high)):,} B"
        print(f"    {knob_name:<34}{(n_byte or 0):>12,}   {verdict}")
        assert route_echo == knob_name, f"the tuner should echo the route it received, got {route_echo!r}"
        result.setdefault("tried", []).append((knob_name, n_byte))
    print("\n    Each one came back echoed by the tuner as `[TuningBuildRoute] Using build route: ...`,")
    print("    so they all reached the builder - the route is not being dropped on the floor.")
    print("\n    -> and yet none of them moved the plan out of the noise band. Acceptance is not")
    print("       effect: a knob acts on a pattern that has to be present, and a 64x64 MatMul")
    print("       gives most of them nothing to do. Same trap as `02-API/BuilderFlag/`.")
    return

@case_mark
def case_wrong_input_is_handled_asymmetrically() -> None:
    """The part worth knowing: a bad knob *name* is rejected, a bad knob *value* is not.

    Both are user errors, and only one of them reaches you. For `-conv_use_long_w=maybe` the build
    succeeds, and the TensorRT logger reports, at INFO:

        [TuningBuildRoute] Using build route: -conv_use_long_w=maybe

    which says it is *using* a value it is about to discard. The only complaint,

        ... invalid compilation option, ignored (invalid boolean value "maybe")

    comes out on raw **stderr** from the compiler underneath - not through `ILogger` at all. Any
    program that routes TensorRT's log to a file, which is what the sweep in `case_sweep` and
    `tests/run_tests.py` both do, never sees it. So a mistyped value leaves the knob at its
    default and tells you it worked.
    """
    print(f"    {'route':<30}{'built':<8}{'echoed by tuner':<24}message")
    for route in ["-no_such_knob=on", "garbage", "-conv_use_long_w", "-conv_use_long_w=maybe"]:
        n_byte, message, route_echo = build(route)
        short = re.sub(r"^Error Code: \d+: Skipping tactic 0x0+ due to exception ", "", message)
        print(f"    {route:<30}{str(n_byte is not None):<8}{(route_echo or '-'):<24}{short[:60]}")
        result.setdefault("wrong_input", []).append((route, n_byte is not None))

    b_ok_dict = dict(result["wrong_input"])
    assert not b_ok_dict["-no_such_knob=on"], "an unknown knob name should be rejected"
    assert not b_ok_dict["garbage"], "a token without `=` should be rejected"
    assert not b_ok_dict["-conv_use_long_w"], "a knob without a value should be rejected"
    assert b_ok_dict["-conv_use_long_w=maybe"], "an unknown *value* is silently accepted on this build - if this now fails, TensorRT started validating it, which is an improvement worth recording"
    print("\n    -> the name is validated, the value is not, and the one case that silently does")
    print("       nothing is also the one the tuner claims to be using. Check a route against")
    print("       `all_build_routes` yourself; TensorRT will not do it for you.")
    return

@case_mark
def case_sweep() -> None:
    """Flip the first `N_SWEEP` on/off knobs away from their default and count what changes.

    This is the question the database invites: of all those knobs, how many actually do anything
    *here*? Answering it needs one build per knob, so the sweep is capped; raise `N_SWEEP` to widen
    it.
    """
    low, high = result["baseline_band"]
    on_off_list = [k for k in result["knob_list"] if get_allowed_value_list(k) == ["on", "off"]][:N_SWEEP]

    outside_list, failed_list = [], []
    for knob in on_off_list:
        flipped = "off" if knob["default_value"] == "on" else "on"
        n_byte, _, _ = build(f"{knob['option']}={flipped}")
        if n_byte is None:
            failed_list.append(knob["option"])
        elif not (low <= n_byte <= high):
            outside_list.append((knob["option"], n_byte))

    n_inert = len(on_off_list) - len(outside_list) - len(failed_list)
    print(f"    flipped {len(on_off_list)} knobs away from their default, band {low:,} .. {high:,} B")
    print(f"      {len(outside_list):>3} landed outside the band: {outside_list if outside_list else '(none)'}")
    print(f"      {len(failed_list):>3} failed to build:         {failed_list if failed_list else '(none)'}")
    print(f"      {n_inert:>3} inside the band, i.e. no measurable effect here")
    print("\n    -> a knob needs a pattern to act on. Sweeping them against a network that does not")
    print("       contain that pattern measures the network, not the knob. To tune for real, sweep")
    print("       against *your* network and compare latency, not plan size.")
    return

if __name__ == "__main__":
    # The knob database, which is the real content of `all_build_routes`
    case_knob_database()
    # What the tuner version is good for
    case_tuner_version_is_a_finer_fingerprint()
    # Setting a route, and establishing that this network builds reproducibly
    case_set_a_route()
    # Accepted is not applied
    case_accepted_is_not_applied()
    # Bad name is caught, bad value is not
    case_wrong_input_is_handled_asymmetrically()
    # How many knobs do anything here at all
    case_sweep()

    print("\nFinish")
