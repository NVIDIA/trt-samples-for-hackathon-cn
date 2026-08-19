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

import re
import subprocess
import sys
from pathlib import Path
from time import time

import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, add_mea, load_mnist_network_trt, case_mark

timing_cache_file = Path("model.TimingCache")
editable_timing_cache_file = Path("model-editable.TimingCache")
b_ignore_mismatch = False  # True allows loading cache created from a different device
shape = [8, 1, 28, 28]

def print_timing_cache_file_info():
    subprocess.run(["ls", "-alh", str(timing_cache_file)], check=False)

def build_network(tw: TRTWrapperV1, network_index: int = 0):
    load_mnist_network_trt(tw)
    tensor = tw.network.get_output(0)

    # Add some extra layers to make the network different
    if network_index == 0:
        tensor = add_mea(tw.network, tensor, [10, 512])
        tensor = add_mea(tw.network, tensor, [512, 10])
    else:
        tensor = add_mea(tw.network, tensor, [10, 768])
        tensor = add_mea(tw.network, tensor, [768, 10])
        tensor = add_mea(tw.network, tensor, [10, 2048])
        tensor = add_mea(tw.network, tensor, [2048, 10])

    layer = tw.network.add_softmax(tensor)
    layer.axes = 1 << 1
    layer = tw.network.add_topk(layer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
    return [layer.get_output(0)]

def load_timing_cache_bytes(cache_file: Path) -> bytes:
    if not cache_file.exists():
        return b""
    with open(cache_file, "rb") as f:
        cache_buffer = f.read()
    print(f"Succeeded loading {cache_file}")
    return cache_buffer

def save_timing_cache(timing_cache: trt.ITimingCache, cache_file: Path):
    with open(cache_file, "wb") as f:
        f.write(timing_cache.serialize())
    print(f"Succeeded saving {cache_file}")

def case_simple(case_index: int, network_index: int, b_use_timing_cache: bool):
    print(f"# Case {case_index} " + "-" * 50)
    tw = TRTWrapperV1()

    timing_cache_buffer = load_timing_cache_bytes(timing_cache_file) if b_use_timing_cache else b""

    if b_use_timing_cache:
        timing_cache = tw.builder_config.create_timing_cache(timing_cache_buffer)
        # timing_cache.reset()  # Reset the timing cache
        tw.builder_config.set_timing_cache(timing_cache, b_ignore_mismatch)

    output_tensor_list = build_network(tw, network_index)

    t0 = time()
    tw.build(output_tensor_list)
    t1 = time()
    print(f"{network_index = }, {b_use_timing_cache = }, build time: {(t1 - t0) * 1000: 10.3f} ms")

    if b_use_timing_cache:
        timing_cache_new = tw.builder_config.get_timing_cache()
        # res = timing_cache.combine(timing_cache_new, b_ignore_mismatch)  # Optional, merge timing cache from the old one (load form file) with the new one (created by this build)
        timing_cache = timing_cache_new
        # print(f"timing_cache.combine: {res}")
        save_timing_cache(timing_cache, timing_cache_file)

PROFILE_BUILD_ARG = "--profile-build"  # Child-process mode, see `profile_build_in_subprocess`

def profile_build_in_subprocess():
    """Build once with a VERBOSE logger in a child process and return its log.

    TensorRT only reveals the *alternative* tactics it evaluated through the verbose build log -
    there is no API to enumerate them. Capturing that log therefore means capturing the whole
    process output, which is why the build runs in a child process: redirecting fd 1/2 in-process
    while TensorRT and CUDA hold them open crashes the runtime.
    """
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), PROFILE_BUILD_ARG],
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout + process.stderr

def parse_tactics(log):
    """Extract `{op_name: (cache_key, selected_tactic_hash, [(cost, tactic_hash), ...])}` from a build log.

    The verbose log emits, per autotuned operator:
        Autotuning op <name>(key: 0x<cache key>): ...
        Sorted table of all evaluated tactics:
        tactic_id, cost(in ms), cost/fastest_cost, prediction_correlation, kernel_name, tactic_hash, ...
           4, 0.0126784, 1.00000, 0.84378, sm80_xmma_gemm_..., 0x2c80ce2f5c4a6,
        ...
        The selected tactic is (tactic hash, cost(in ms)):0x2c80ce2f5c4a6, 0.0126784
    """
    result = {}
    for name, key, body in re.findall(r"Autotuning op (\S+?)\(key: (0x[0-9a-f]+)\)(.*?)(?=Autotuning op |\Z)", log, re.S):
        tactic_list = re.findall(r"^.*?\s\d+, ([\d.]+), [\d.]+, [\d.]+, \S+?, (0x[0-9a-f]+),", body, re.M)
        selected = re.search(r"The selected tactic is \(tactic hash, cost\(in ms\)\):(0x[0-9a-f]+)", body)
        if tactic_list and selected:
            result[name] = (key, selected.group(1), tactic_list)
    return result

@case_mark
def case_force_tactic():
    """Force a deliberately non-fastest tactic, so the build becomes reproducible.

    TensorRT profiles every tactic of a layer and keeps the fastest. When a different one is wanted
    - to reproduce a known-good build, to dodge a tactic that misbehaves on some input, or just to
    make builds deterministic - the choice can be overridden by writing that tactic into the
    editable timing cache before the build reads it.
    """
    log = profile_build_in_subprocess()
    tactic_dict = parse_tactics(log)
    print(f"Autotuned operators with more than one tactic: {sum(len(v[2]) > 1 for v in tactic_dict.values())} / {len(tactic_dict)}")
    if not tactic_dict:
        print("No autotuning information in the build log, skip forcing a tactic")
        return

    # Pick any operator that really had a choice, and take the runner-up rather than the winner.
    name, (key_string, selected_hash, tactic_list) = next(((k, v) for k, v in tactic_dict.items() if len(v[2]) > 1), (None, None))
    if name is None:
        print("Every operator had exactly one tactic, nothing to force")
        return
    cost, forced_hash = next((c, h) for c, h in tactic_list if h != selected_hash)
    print(f"Operator {name}: TensorRT picked {selected_hash}, forcing {forced_hash} (cost {cost} ms) instead")

    # Rebuild with the cache entry overwritten. `TimingCacheKey.parse` turns the key printed in the
    # log back into the key object the cache is indexed by.
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    timing_cache = tw.builder_config.create_timing_cache(load_timing_cache_bytes(timing_cache_file))
    key = trt.TimingCacheKey.parse(key_string)
    print(f"Cache update accepted: {timing_cache.update(key, trt.TimingCacheValue(int(forced_hash, 16), float(cost)))}")
    tw.builder_config.set_timing_cache(timing_cache, b_ignore_mismatch)
    tw.build(build_network(tw, 0))

    # If TensorRT had re-profiled this operator it would have written the fastest tactic back into
    # the cache. Finding the forced one still there means the build consumed the edited entry
    # instead of measuring again.
    final_value = tw.builder_config.get_timing_cache().query(key)
    print(f"Tactic in the cache after the build: {hex(final_value.tacticHash)}")
    print(f"Forced tactic survived the build: {hex(final_value.tacticHash) == forced_hash}")

@case_mark
def case_editable():

    # Build a baseline timing cache
    tw = TRTWrapperV1()
    output_tensor_list = build_network(tw, 0)

    timing_cache_buffer = b""
    timing_cache = tw.builder_config.create_timing_cache(timing_cache_buffer)
    tw.builder_config.set_timing_cache(timing_cache, b_ignore_mismatch)

    tw.build(output_tensor_list)

    timing_cache = tw.builder_config.get_timing_cache()
    save_timing_cache(timing_cache, timing_cache_file)

    # Edit the timing cache
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)

    timing_cache = tw.builder_config.create_timing_cache(load_timing_cache_bytes(timing_cache_file))
    key_list = timing_cache.queryKeys()
    if len(key_list) == 0:
        print("No key in timing cache, skip editing")
        return

    # Each element returned by `queryKeys()` is a `trt.TimingCacheKey`, identifying one layer /
    # tactic entry in the editable timing cache. A key can also be reconstructed from its string
    # form with the static method `trt.TimingCacheKey.parse(...)`.
    key: trt.TimingCacheKey = key_list[0]
    print(f"Type of timing cache key: {type(key)}")
    old_value = timing_cache.query(key)

    print(f"Old cache value: tacticHash={old_value.tacticHash}, timingMSec={old_value.timingMSec:.6f}")

    new_value = trt.TimingCacheValue(int(old_value.tacticHash), max(float(old_value.timingMSec) * 0.8, 1e-6))
    status = timing_cache.update(key, new_value)
    print(f"Timing cache update status: {status}")

    check_value = timing_cache.query(key)
    print(f"New cache value: tacticHash={check_value.tacticHash}, timingMSec={check_value.timingMSec:.6f}")

    save_timing_cache(timing_cache, editable_timing_cache_file)

    # Build engine (of different network) with the edited timing cache
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)

    timing_cache = tw.builder_config.create_timing_cache(load_timing_cache_bytes(editable_timing_cache_file))
    tw.builder_config.set_timing_cache(timing_cache, b_ignore_mismatch)

    output_tensor_list = build_network(tw, 1)
    tw.build(output_tensor_list)

#######################

if __name__ == "__main__":
    if PROFILE_BUILD_ARG in sys.argv:
        # Child-process mode of `case_force_tactic`: build once with a VERBOSE logger so the parent
        # can read the per-operator tactic tables out of this process's output, and leave the
        # resulting cache on disk for the parent to edit.
        tw_child = TRTWrapperV1(logger="verbose")
        tw_child.builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
        cache_child = tw_child.builder_config.create_timing_cache(b"")
        tw_child.builder_config.set_timing_cache(cache_child, b_ignore_mismatch)
        tw_child.build(build_network(tw_child, 0))
        save_timing_cache(tw_child.builder_config.get_timing_cache(), timing_cache_file)
        sys.exit(0)

    timing_cache_file.unlink(missing_ok=True)
    editable_timing_cache_file.unlink(missing_ok=True)

    # Case 0, Build network 0 without timing cache
    case_simple(0, 0, 0)

    # Case 1, Build network 0 again without timing cache, build-time is a little bit shorter than Case 0 due to GPU warming up
    case_simple(1, 0, 0)

    # Case 2, Build network 1 without timing cache
    case_simple(2, 1, 0)

    # Case 3, Build network 1 again without timing cache, build-time is similar to Case 2
    case_simple(3, 1, 0)

    # Case 4, Build network 0 with writing timing cache, build-time is similar to Case 1
    case_simple(4, 0, 1)
    print_timing_cache_file_info()

    # Case 5, Build network 0 again with reading timing cache, build time is much shorter than Case 4
    case_simple(5, 0, 1)
    print_timing_cache_file_info()

    # Case 6, Build network 1 with reading and appending timing cache, build-time is somehow shorter than Case 3
    # i.e. timing cache can be used from a similar but different network.
    # Meanwhile, the size of file `model.TimingCache` increases
    case_simple(6, 1, 1)
    print_timing_cache_file_info()

    # Case 7, Build network 1 again with reading timing cache, build-time is much shorter than Case 6
    case_simple(7, 1, 1)
    print_timing_cache_file_info()

    # Case 8, Build network 0 again with reading timing cache, build-time is similar to Case 5
    # i.e. timing cache of both network 0 and 1 are stored together
    case_simple(8, 0, 1)
    print_timing_cache_file_info()

    # Case 9, use editable timing cache
    case_editable()

    # Case 10, force a specific (non-fastest) tactic through the editable timing cache
    case_force_tactic()

    print("Finish")
