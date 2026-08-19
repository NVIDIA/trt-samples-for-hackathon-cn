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

from pathlib import Path

import tensorrt as trt

from tensorrt_cookbook import (TRTWrapperV1, build_mnist_network_trt, case_mark, print_engine_io_information)

# Reading the plan header requires knowledge of TensorRT's serialized layout, which is not public,
# so that half of this example ships separately and is normally absent. Everything below still runs
# without it; this is the only thing the rest of the file needs to know.
try:
    from tensorrt_cookbook import parse_engine_information, print_engine_information
    B_HAS_PLAN_PARSER = True
except ImportError:
    B_HAS_PLAN_PARSER = False

output_path = Path(__file__).parent
trt_file = output_path / "model-trained.trt"

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw)

    # `build_mnist_network_trt` configures `tw.profile` (batch 1 / 2 / 4) but does not add it to the
    # builder config; `tw.build` adds that one first and then any extras, so this becomes profile 1
    profile_large_batch = tw.builder.create_optimization_profile()
    profile_large_batch.set_shape("x", [8, 1, 28, 28], [32, 1, 28, 28], [64, 1, 28, 28])

    tw.build(output_tensor_list, extra_profile_list=[profile_large_batch])
    tw.serialize_engine(trt_file)
    print(f"    Built {trt_file.name}, {trt_file.stat().st_size / (1 << 20):.2f} MiB")

    # Engine metadata read out of the serialized bytes: which TensorRT built the plan, its hardware
    # compatibility level, and the device it records having been built for.
    if B_HAS_PLAN_PARSER:
        print_engine_information(trt_file=trt_file, plugin_file_list=[], device_index=0)
    else:
        print("    [SKIP] `print_engine_information` is not available in this distribution.")
        print("           It reads the plan header field by field, and the serialized layout of a")
        print("           TensorRT engine is not part of the public API, so the tool is not shipped.")

    # Input / output tensors, with the shape range of every optimization profile. Public API only,
    # so this half is always available.
    print_engine_io_information(trt_file=trt_file, plugin_file_list=[])

@case_mark
def case_check_against_the_api():
    """See the docstring below; skipped without the plan parser."""
    if not B_HAS_PLAN_PARSER:
        print("    [SKIP] needs `parse_engine_information`, which is not shipped with this cookbook.")
        return
    return _check_against_the_api()

def _check_against_the_api():
    """Assert on the parsed values, which is why the parse is separate from the print.

    Every byte offset in the parser is specific to one TensorRT layout, and when a layout changes
    the output does not look broken - it looks plausible. The 10.x version of this code, run on
    11.0, reported a TensorRT version of `0.0.0.0` and a 422-terabyte archive, and nothing
    complained, because a human reading a log was the only consumer.

    These four assertions are cheap and each one would have failed on the day 11.0 arrived.
    """
    info = parse_engine_information(trt_file)

    # The plan says which TensorRT built it, and this one was built moments ago by the installed one
    assert info["plan"]["trtVersion"] == trt.__version__, \
        f"plan header says {info['plan']['trtVersion']}, installed TensorRT is {trt.__version__}"
    # The engine archive repeats the version; the two must agree
    assert info["archive"]["trtVersion"] == info["plan"]["trtVersion"]
    # `kENGINE` is an archive, `kWEIGHTS` is a raw blob - checked rather than assumed
    type_to_archive = {e["typeName"]: e["isArchive"] for e in info["entry"]}
    assert type_to_archive["kENGINE"] is True and type_to_archive["kWEIGHTS"] is False, type_to_archive
    # The device block has to be the device we are on, since we just built the engine here
    assert info["device"] == info["deviceCurrent"], "engine and current device disagree"

    n_failed = sum(1 for c in info["check"] if not c["ok"])
    print(f"    {len(info['check'])} structural checks, {n_failed} failed")
    print(f"    plan built by TensorRT {info['plan']['trtVersion']}, running {trt.__version__}")
    print("    All four assertions hold; a layout change would break them rather than print nonsense.")

if __name__ == "__main__":
    case_simple()
    case_check_against_the_api()

    print("Finish")
