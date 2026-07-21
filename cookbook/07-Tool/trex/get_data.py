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

# Generate the TensorRT engine-plan JSON files that the trex examples consume.
#
# This mirrors the classic `trt-engine-explorer` workflow: use `trtexec` to
#   1. build an engine and export its graph JSON (detailed profiling verbosity),
#   2. profile the engine and export the per-layer profile + per-iteration timing.
# The resulting JSON files are GPU-free to analyse afterwards.
#
# Two engines are built from the same MNIST network so the CompareEngines example
# has something to compare:
#   model.*       - an INT8 (QAT) engine
#   model.fp16.*  - an FP16 engine
# All other examples only use the INT8 engine (`model.*`).

import subprocess
from pathlib import Path

model_dir = Path("/work/trt-samples-for-hackathon-cn/cookbook/00-Data/model")
data_path = Path(__file__).parent / "data"

# Build with a fixed shape (min == opt == max) so the engine-graph JSON reports
# concrete tensor dimensions; this keeps the activation byte accounting correct.
# (Dynamic shape profiles are covered in a dedicated trex example.)
shapes = ["--minShapes=x:4x1x28x28", "--optShapes=x:4x1x28x28", "--maxShapes=x:4x1x28x28"]

# (json prefix, onnx file, precision flag)
engines = [
    ("model", model_dir / "model-trained-int8-qat.onnx", "--int8"),
    ("model.fp16", model_dir / "model-trained.onnx", "--fp16"),
]

def run(cmd, log_file):
    print(f"[get_data] Running: {' '.join(cmd)}")
    with open(log_file, "w") as f:
        completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed (exit {completed.returncode}), see {log_file}")

def build_and_profile(prefix, onnx_file, precision_flag):
    engine_file = data_path / f"{prefix}.engine"
    graph_json = data_path / f"{prefix}.graph.json"
    profile_json = data_path / f"{prefix}.profile.json"
    timing_json = data_path / f"{prefix}.timing.json"

    # The JSON files are shared by all trex sub-examples; skip if already built.
    if graph_json.exists() and profile_json.exists() and timing_json.exists():
        print(f"[get_data] Reusing existing JSON files for '{prefix}'")
        return

    # 1. Build the engine and export the engine-graph JSON.
    run(
        [
            "trtexec",
            f"--onnx={onnx_file}",
            f"--saveEngine={engine_file}",
            precision_flag,
            "--profilingVerbosity=detailed",  # required for a detailed graph JSON
            f"--exportLayerInfo={graph_json}",
            "--skipInference",
            *shapes,
        ],
        data_path / f"{prefix}.build.log",
    )

    # 2. Profile the engine and export the profile + timing JSON.
    run(
        [
            "trtexec",
            f"--loadEngine={engine_file}",
            f"--exportProfile={profile_json}",
            f"--exportTimes={timing_json}",
            "--dumpProfile",
            # Measure e2e timing in a separate run; otherwise the profiler's extra
            # synchronizations perturb it and trtexec skips the timing export.
            "--separateProfileRun",
            "--shapes=x:4x1x28x28",
            "--iterations=50",
            "--warmUp=200",
        ],
        data_path / f"{prefix}.profile.log",
    )
    print(f"[get_data] Wrote {graph_json}, {profile_json}, {timing_json}")

def main():
    data_path.mkdir(exist_ok=True)
    for prefix, onnx_file, precision_flag in engines:
        build_and_profile(prefix, onnx_file, precision_flag)

if __name__ == "__main__":
    main()
