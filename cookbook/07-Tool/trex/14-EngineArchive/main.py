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

# Bundle an engine and its analysis artifacts into a TensorRT Engine Archive (TEA).
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# archiving.EngineArchive. A TEA is a ZIP file bundling the engine plan together
# with its graph/profile JSON and a plan-info JSON (extracted by deserializing
# the engine), so a whole exploration session travels as one file.
#
# Deserializing the engine for plan-info requires TensorRT (a GPU is not needed
# just to deserialize, but the engine must match this TensorRT version).

import json
from pathlib import Path

from tensorrt_cookbook import EngineArchive, case_mark

data_path = Path(__file__).parent.parent / "data"  # shared engine + JSON files
out_path = Path(__file__).parent
engine_file = data_path / "model.engine"
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"
tea_file = out_path / "model.engine.tea"

@case_mark
def case_create_archive():
    """Bundle the engine + JSON artifacts + plan info into a .tea archive."""
    plan_bytes = engine_file.read_bytes()
    with EngineArchive(str(tea_file), mode="w") as tea:
        tea.writef_bin("engine.trt", plan_bytes)
        tea.add_file(str(graph_json), "model.graph.json")
        tea.add_file(str(profile_json), "model.profile.json")
        # Deserialize the engine and archive its properties + IO tensors.
        tea.archive_plan_info(plan_bytes)
        print(f"Archived entries: {tea.namelist()}")
    print(f"Wrote {tea_file.name} ({tea_file.stat().st_size / 1024:.1f} KB)")

@case_mark
def case_read_archive():
    """Read the archive back and show the plan-info it stored."""
    with EngineArchive(str(tea_file), mode="r") as tea:
        print(f"Entries: {tea.namelist()}")
        plan_info = json.loads(tea.readf("plan_cfg.json"))

    print(f"\nEngine name       : {plan_info.get('name')}")
    print(f"Num layers        : {plan_info.get('num_layers')}")
    print(f"Num optim profiles: {plan_info.get('num_optimization_profiles')}")
    print("IO tensors:")
    for name, info in plan_info["io_tensors"].items():
        print(f"    {name}: {info['mode']} {info['dtype']} {info['shape']}")

if __name__ == "__main__":
    case_create_archive()
    case_read_archive()

    print("Finish")
