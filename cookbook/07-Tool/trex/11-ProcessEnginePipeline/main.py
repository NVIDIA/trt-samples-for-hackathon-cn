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

# The full trex workflow, end to end: ONNX -> build -> profile -> JSON -> explore.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# `utils/process_engine.py`, which drives `trtexec` to build and profile an
# engine and generate all the JSON artifacts trex consumes. Unlike the other
# trex examples (which share one prebuilt engine), this example builds its own
# engine, so it REQUIRES A GPU and `trtexec`.
#
# It ties together the pieces from the other examples:
#   build + profile (trtexec)  ->  graph/profile/timing JSON
#   parse logs (#09)           ->  build/profile metadata JSON
#   render graph (#02)         ->  engine SVG
#   EnginePlan + summary (#00) ->  textual report

import subprocess
from pathlib import Path

from tensorrt_cookbook import (
    EnginePlan,
    case_mark,
    print_precision_stats,
    print_summary,
    render_engine_graph,
    write_build_metadata,
    write_profiling_metadata,
)

onnx_file = Path("/work/trt-samples-for-hackathon-cn/cookbook/00-Data/model/model-trained.onnx")
out_dir = Path(__file__).parent / "pipeline_out"
name = "model-trained"

engine_file = out_dir / f"{name}.engine"
graph_json = out_dir / f"{name}.graph.json"
profile_json = out_dir / f"{name}.profile.json"
timing_json = out_dir / f"{name}.timing.json"
build_log = out_dir / f"{name}.build.log"
profile_log = out_dir / f"{name}.profile.log"
build_meta = out_dir / f"{name}.build.metadata.json"
profile_meta = out_dir / f"{name}.profile.metadata.json"

shapes = ["--minShapes=x:4x1x28x28", "--optShapes=x:4x1x28x28", "--maxShapes=x:4x1x28x28"]

def _run(cmd, log_file):
    print(f"    $ {' '.join(cmd)}")
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"trtexec failed (exit {rc}); see {log_file}")

@case_mark
def case_build():
    """Step 1: build the engine and export the engine-graph JSON."""
    out_dir.mkdir(exist_ok=True)
    _run(
        ["trtexec", f"--onnx={onnx_file}", f"--saveEngine={engine_file}", "--fp16", "--profilingVerbosity=detailed", f"--exportLayerInfo={graph_json}", "--skipInference", *shapes],
        build_log,
    )

@case_mark
def case_profile():
    """Step 2: profile the engine and export the profile + timing JSON."""
    _run(
        ["trtexec", f"--loadEngine={engine_file}", f"--exportProfile={profile_json}", f"--exportTimes={timing_json}", "--dumpProfile", "--separateProfileRun", "--shapes=x:4x1x28x28", "--iterations=50", "--warmUp=200"],
        profile_log,
    )

@case_mark
def case_metadata():
    """Step 3: parse the trtexec logs into metadata JSON."""
    write_build_metadata(str(build_log), str(build_meta))
    write_profiling_metadata(str(profile_log), str(profile_meta))
    print(f"    wrote {build_meta.name}, {profile_meta.name}")

@case_mark
def case_draw():
    """Step 4: render the engine graph to SVG."""
    plan = EnginePlan(str(graph_json), str(profile_json), name=name)
    out = render_engine_graph(plan, out_dir / f"{name}.graph", "svg")
    print(f"    wrote {Path(out).name}")

@case_mark
def case_explore():
    """Step 5: load everything into an EnginePlan and print the report."""
    plan = EnginePlan(
        str(graph_json),
        str(profile_json),
        profiling_metadata_file=str(profile_meta),
        build_metadata_file=str(build_meta),
        name=name,
    )
    print_summary(plan)
    print_precision_stats(plan)

if __name__ == "__main__":
    case_build()
    case_profile()
    case_metadata()
    case_draw()
    case_explore()

    print("Finish")
