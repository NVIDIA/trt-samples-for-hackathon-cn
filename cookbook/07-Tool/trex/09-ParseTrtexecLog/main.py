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

# Parse trtexec build / profiling logs into the metadata JSON files that carry
# the device properties, builder configuration and performance summary.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# `utils/parse_trtexec_log.py`. Feeding the resulting metadata back into
# EnginePlan fills the Device / Builder / Performance sections of print_summary
# (which are empty when only the graph + profile JSON are provided).

from pathlib import Path

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    parse_build_log,
    parse_profiling_log,
    print_summary,
    write_build_metadata,
    write_profiling_metadata,
)

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON + logs
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"
build_log = data_path / "model.build.log"
profile_log = data_path / "model.profile.log"
# Write the metadata next to this example (do not touch the shared data/ dir).
out_path = Path(__file__).parent
build_meta = out_path / "model.build.metadata.json"
profile_meta = out_path / "model.profile.metadata.json"

@case_mark
def case_parse_logs():
    """Parse the build/profile logs and print the extracted key fields."""
    build = parse_build_log(str(build_log))
    profile = parse_profiling_log(str(profile_log))

    print("Device information (from build log):")
    for k in ("Selected Device", "Compute Capability", "SMs", "Device Global Memory"):
        print(f"    {k}: {build['device_information'].get(k)}")

    print("\nBuild options (subset):")
    for k in ("Precision", "Int8", "TF32"):
        if k in build["build_options"]:
            print(f"    {k}: {build['build_options'][k]}")

    print("\nPerformance summary (from profile log):")
    for k in ("Throughput", "Latency", "GPU Compute Time"):
        print(f"    {k}: {profile['performance_summary'].get(k)}")

@case_mark
def case_write_metadata():
    """Write the metadata JSON files that EnginePlan consumes."""
    write_build_metadata(str(build_log), str(build_meta))
    write_profiling_metadata(str(profile_log), str(profile_meta))
    print(f"Wrote {build_meta.name} and {profile_meta.name}")

@case_mark
def case_summary_with_metadata():
    """Load the plan WITH metadata: the summary now shows device / builder / perf."""
    plan = EnginePlan(
        str(graph_json),
        str(profile_json),
        profiling_metadata_file=str(profile_meta),
        build_metadata_file=str(build_meta),
        name="model",
    )
    print_summary(plan)

if __name__ == "__main__":
    case_parse_logs()
    case_write_metadata()
    case_summary_with_metadata()

    print("Finish")
