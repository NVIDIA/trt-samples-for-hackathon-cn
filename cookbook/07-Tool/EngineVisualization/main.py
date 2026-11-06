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

import json
import subprocess
from pathlib import Path

from tensorrt_cookbook import (build_cyclic_reference_onnx, case_mark, compare_engine_profiles, cookbook_path, draw_engine_graph, export_engine_graph_as_onnx)

output_path = Path(__file__).parent
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
json_file = output_path / "data-layer_info.json"
profile_file = output_path / "data-profile.json"

# Pictures
graphviz_file = output_path / "result-engine.gv"
image_file = output_path / "result-engine.svg"
html_file = output_path / "result-engine.html"
graphml_file = output_path / "result-engine.graphml"
simple_graphviz_file = output_path / "result-engine-simple.gv"

multi_profile_json_file = output_path / "data-layer_info-multi_profile.json"
multi_profile_graphviz_file = output_path / "result-engine-multi_profile.gv"
multi_profile_image_file = output_path / "result-engine-multi_profile.svg"

# ONNX, which is a real graph object rather than a picture
export_json_file = output_path / "data-layer_info-export.json"
export_file = output_path / "result-engine.onnx"
export_file_no_profile = output_path / "result-engine-no_profile.onnx"
cyclic_reference_file = output_path / "reference-cyclic-no-subgraph.onnx"

def run_trtexec(argument_list: list) -> None:
    process = subprocess.run(["trtexec", *argument_list], capture_output=True, text=True, cwd=str(output_path))
    assert process.returncode == 0, process.stdout[-1500:]

@case_mark
def case_draw_every_format() -> None:
    # Build TensorRT engine with dumped data
    # `--exportLayerInfo` in trtexec is equivalent to `IEngineInspector` in Python API
    # trtexec is used here because we want `--exportProfile` for performance data (optional for the visualization)
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={json_file}",
        f"--exportProfile={profile_file}",  # Optional, this will put a latency data on every node
        "--profilingVerbosity=detailed",  # Without this the names are useless
        "--iterations=10",  # `--exportProfile` needs the engine to actually run
    ])
    assert json_file.exists()
    data = json.loads(json_file.read_text())
    print(f"    {json_file.name}: {len(data['Layers'])} layers, top-level keys {list(data)}")

    # Read the JSON, draw every format, and check what was drawn. An output is produced only if its
    # path is given, so leaving one out is how you ask for fewer files.
    draw_engine_graph(
        json_file,
        profile_file,  # Optional, this is what puts a latency on every node
        graphviz_file=graphviz_file,
        image_file=image_file,
        html_file=html_file,
        graphml_file=graphml_file,
    )
    print("    Edges are implicit in the JSON: A -> B when a name in A's Outputs is in B's Inputs.")
    print("    A tensor may have several producers - keeping only the last would drop the loop edges.")

    # The same graph with everything optional turned off, which is what stays readable on a large
    # engine. `b_simplify` is a property of the drawing, so both versions are two calls, not one.
    draw_engine_graph(json_file, profile_file, graphviz_file=simple_graphviz_file, b_simplify=True, b_print=False)
    print(f"    {simple_graphviz_file.name}: {simple_graphviz_file.stat().st_size} B "
          f"(vs {graphviz_file.stat().st_size} B with labels, bindings, latency and full tooltips)")
    return

@case_mark
def case_multi_profile() -> None:
    """
    `load_engine_graph` returns only profile 0 for the engien with multiple optimization profilers.
    """
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={multi_profile_json_file}",
        "--profilingVerbosity=detailed",
        "--skipInference",  # No latency wanted here, so the engine never has to run
        "--profile=0",
        "--minShapes=x:1x1x28x28",
        "--optShapes=x:2x1x28x28",
        "--maxShapes=x:4x1x28x28",
        "--profile=1",
        "--minShapes=x:8x1x28x28",
        "--optShapes=x:32x1x28x28",
        "--maxShapes=x:64x1x28x28",
    ])
    # Split the layer list by profile and report how the copies differ, without building graphs
    # nobody is going to draw
    report = compare_engine_profiles(multi_profile_json_file)
    assert len({len(item) for item in report["layers_per_profile"]}) == 1, "the profiles should hold one copy of the network each"

    # Draw profile 0, which is all `draw_engine_graph` builds. No `html_file` or `graphml_file`
    # here, so those two formats are simply not written
    graph = draw_engine_graph(
        multi_profile_json_file,
        graphviz_file=multi_profile_graphviz_file,
        image_file=multi_profile_image_file,
    )
    print(f"    profile 0 only: {len(graph.node_list)} of {report['n_layer']} layers drawn")
    return

@case_mark
def case_export_as_onnx() -> None:
    """The same layer-info JSON as an ONNX file, which Netron opens and ONNX tooling can read.

    A different kind of conversion from the pictures above: those format a node and edge list as
    text, this builds a real `onnx_graphsurgeon` graph.

    Passing `profile_file` is what fills in the `Latency` attribute. The **attribute set does not
    depend on it** - without a profile the attribute is there and empty - so a reader never has to
    handle two shapes of file. That is asserted below rather than taken on trust.
    """
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={export_json_file}",
        f"--exportProfile={profile_file}",
        "--profilingVerbosity=detailed",  # Without this the layer names are placeholders
        "--iterations=10",  # `--exportProfile` needs the engine to actually run
    ])
    report = export_engine_graph_as_onnx(export_json_file, profile_file, onnx_file=export_file)

    without_profile = export_engine_graph_as_onnx(export_json_file, onnx_file=export_file_no_profile, b_print=False)
    assert without_profile["attribute_name_set"] == report["attribute_name_set"], "the attribute set changed with the profile"
    assert without_profile["latency_sample"] is None and report["latency_sample"] is not None
    print(f"    exported again without the profile: same {len(report['attribute_name_set'])} attributes, Latency now empty")
    return

@case_mark
def case_export_onnx_corpus() -> None:
    """A spread of engines, so the edge cases stay covered rather than being rediscovered.

    Each of these has broken the exporter at some point: `model-trained-int8-qat` carries Q/DQ
    pairs, `model-labeled` has named dimensions, and `model-redundant` collapses to a single layer.
    The check is `onnx.checker`, not merely "did not raise" - a graph only Netron will open is not
    good enough, because the point of writing ONNX is that other ONNX tooling can read it.
    """
    for model_name in ["model-trained-int8-qat", "model-labeled", "model-redundant"]:
        corpus_json_file = output_path / f"data-layer_info-{model_name}.json"
        run_trtexec([
            f"--onnx={cookbook_path('00-Data', 'model', f'{model_name}.onnx')}",
            f"--exportLayerInfo={corpus_json_file}",
            "--profilingVerbosity=detailed",
            "--builderOptimizationLevel=0",
            "--skipInference",
        ])
        report = export_engine_graph_as_onnx(corpus_json_file, onnx_file=output_path / f"result-engine-{model_name}.onnx", b_print=False)
        print(f"    {model_name:22s}: {report['n_node']:3d} nodes, onnx.checker passed")
    return

@case_mark
def case_export_onnx_control_flow() -> None:
    """An engine containing a `Loop`, exported both ways.

    TensorRT compiles a `Loop` into a flat instruction stream with a back-edge, so a tensor name has
    several producers and one layer reads and writes the same name.

    + `onnx_file` keeps those names, so the file has the cycle in it and a viewer draws the loop.
    + `dag_onnx_file` renames every write to an SSA version and adds a `BackEdge` marker node per
      chain. The result is a DAG that passes `onnx.checker`, which is what ONNX tooling such as
      `onnx_outliner` needs - at the cost of being harder to read, since the back-edge becomes a
      node whose target you have to look up by name.

    The picture writers do neither: they keep every producer and mark the back-edges dashed.
    """
    build_cyclic_reference_onnx(cyclic_reference_file)

    for model_name, description in [("model-loop", "a Loop and nothing else"), ("model-for", "a Loop with an If in its body")]:
        control_flow_json_file = output_path / f"data-layer_info-{model_name}.json"
        run_trtexec([
            f"--onnx={cookbook_path('00-Data', 'model', f'{model_name}.onnx')}",
            f"--exportLayerInfo={control_flow_json_file}",
            "--profilingVerbosity=detailed",
            "--skipInference",
        ])
        print(f"    {model_name} ({description})")
        report = export_engine_graph_as_onnx(
            control_flow_json_file,
            onnx_file=output_path / f"result-engine-{model_name}.onnx",
            dag_onnx_file=output_path / f"result-engine-{model_name}-dag.onnx",
            b_print=False,
        )
        assert report["n_cycle"] > 0, f"{model_name} contains a Loop but the default export has no cycle"
        print(f"        default         : {report['n_node']:3d} nodes, {report['n_cycle']} back-edge(s) left as cycles, names unchanged")
        print(f"        b_break_cycle   : {report['n_back_edge']} BackEdge marker(s), onnx.checker passed")
    return

if __name__ == "__main__":
    case_draw_every_format()
    case_multi_profile()

    case_export_as_onnx()
    case_export_onnx_corpus()
    case_export_onnx_control_flow()

    print("\nFinish")
