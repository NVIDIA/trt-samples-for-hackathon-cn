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

# Export a TensorRT engine plan to an ONNX file for viewing in Netron.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# graphing.OnnxGraph / make_onnx_tensor. One ONNX node is emitted per engine
# layer, wired by tensor names; the engine bindings become the ONNX graph I/O.
#
# NOTE: the exported ONNX is a *visualization aid*, not a runnable model - it
# uses TensorRT layer names (e.g. "Reformat") as op types, which Netron shows
# fine but `onnx.checker` / onnxruntime will reject.

from pathlib import Path

import onnx

from tensorrt_cookbook import EnginePlan, case_mark, export_engine_to_onnx

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own output
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"
onnx_file = out_path / "engine_model.onnx"

@case_mark
def case_export_onnx():
    """Export the engine plan to ONNX and report the resulting graph."""
    plan = EnginePlan(str(graph_json), str(profile_json), name="model")
    out = export_engine_to_onnx(plan, str(onnx_file))
    print(f"Exported engine to {out}")

    # Load it back and describe the graph (do NOT run onnx.checker: the op types
    # are TensorRT layer names, not standard ONNX ops).
    model = onnx.load(out)
    graph = model.graph
    print(f"Nodes   = {len(graph.node)}")
    print(f"Inputs  = {[i.name for i in graph.input]}")
    print(f"Outputs = {[o.name for o in graph.output]}")

    print("\nFirst few nodes (op_type: name):")
    for node in graph.node[:5]:
        print(f"    {node.op_type}: {node.name}")

    print("\nOpen the file in Netron to explore the engine graph:")
    print("    netron engine_model.onnx")

if __name__ == "__main__":
    case_export_onnx()

    print("Finish")
