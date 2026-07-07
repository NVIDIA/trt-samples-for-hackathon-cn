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

# Draw a TensorRT engine plan as a graph (SVG / PNG) with Graphviz.
#
# This is the cookbook re-implementation of `trt-engine-explorer`'s
# graphing.DotGraph / to_dot / render_dot and the `utils/draw_engine.py` script.
# Layer nodes are colored by layer type, edges by tensor precision. Requires the
# Graphviz `dot` binary (`apt-get install graphviz`).

from pathlib import Path

import numpy as np

from tensorrt_cookbook import EnginePlan, case_mark, render_engine_graph

data_path = Path(__file__).parent.parent / "data"  # shared engine JSON files
out_path = Path(__file__).parent  # this example's own figures
graph_json = data_path / "model.graph.json"
profile_json = data_path / "model.profile.json"

# Output format for the rendered graph ("svg" is compact and zoomable; "png"
# is easy to preview). Change and re-run to taste.
output_format = "png"

def _load():
    return EnginePlan(str(graph_json), str(profile_json), name="model")

@case_mark
def case_draw_default():
    """Render the full engine graph: layers by type, edges by precision."""
    plan = _load()
    out = render_engine_graph(plan, out_path / "engine_graph", output_format)
    print(f"Saved {out}")

@case_mark
def case_draw_svg():
    """The same graph as SVG (vector, good for zooming into large engines)."""
    plan = _load()
    out = render_engine_graph(plan, out_path / "engine_graph", "svg")
    print(f"Saved {out}")

@case_mark
def case_draw_simplified():
    """A simplified graph: no edge shape/dtype labels, no binding nodes."""
    plan = _load()
    out = render_engine_graph(
        plan,
        out_path / "engine_graph_simple",
        output_format,
        display_edge_details=False,
        display_bindings=False,
        display_latency=False,
    )
    print(f"Saved {out}")

@case_mark
def case_draw_highlight():
    """Highlight the slowest layer with a red border."""
    plan = _load()
    avg = plan.col("latency.avg_time").astype(float)
    slowest = plan.records[int(np.argmax(avg))]["Name"]
    print(f"Highlighting slowest layer: {slowest}")
    out = render_engine_graph(
        plan,
        out_path / "engine_graph_highlight",
        output_format,
        highlight_layers=[slowest],
    )
    print(f"Saved {out}")

if __name__ == "__main__":
    case_draw_default()
    case_draw_svg()
    case_draw_simplified()
    case_draw_highlight()

    print("Finish")
