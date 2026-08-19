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
"""The fourth output format for the same layer-info JSON: an ONNX file, opened with Netron.

`main.py` writes `.gv`, `.html` and `.graphml` from an engine's layer-info JSON. This writes the
same graph as ONNX, via `export_engine_as_onnx`. It is a separate script rather than another case in
`main.py` because it is a different kind of conversion: the other three build a node and edge list
and format it, while this one builds a real `onnx_graphsurgeon` graph and therefore keeps everything
the JSON says about each layer and each tensor.

See `README.md` for what to reach for when. The short version: this format keeps the most
information and is the one other ONNX tooling can consume; the other three are for looking at the
graph without Netron, or for editing its layout by hand.

```bash
python3 export_as_onnx.py
```
"""

import subprocess
from collections import OrderedDict
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tensorrt_cookbook import case_mark, cookbook_path, export_engine_as_onnx

output_path = Path(__file__).parent
result = OrderedDict()

# The same MNIST engine `main.py` uses, so the two sets of outputs describe the same graph - but
# written to its own file, so neither script depends on having run after the other
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
json_file = output_path / "data-layer_info-export.json"
profile_file = output_path / "data-profile.json"
export_file = output_path / "result-engine.onnx"
cyclic_reference_file = output_path / "reference-cyclic-no-subgraph.onnx"

def run_trtexec(argument_list: list) -> None:
    process = subprocess.run(["trtexec", *argument_list], capture_output=True, text=True, cwd=str(output_path))
    assert process.returncode == 0, process.stdout[-1500:]

@case_mark
def case_export() -> None:
    """Layer-info JSON -> ONNX, and what survives the trip."""
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={json_file.name}",
        "--profilingVerbosity=detailed",  # Without this the layer names are placeholders
        "--skipInference",
    ])
    export_engine_as_onnx(engine_json_file=json_file, export_onnx_file=export_file)

    model = onnx.load(export_file)
    onnx.checker.check_model(model)
    attribute_name_set = sorted({attribute.name for node in model.graph.node for attribute in node.attribute})
    print(f"    {len(model.graph.node)} nodes, node attributes kept: {attribute_name_set}")
    print(f"    opset imports: {[(opset.domain or '<default>', opset.version) for opset in model.opset_import]}")
    print("    `onnx.checker` passes, so tooling other than Netron will also read this file.")
    result["attribute_name_set"] = attribute_name_set

@case_mark
def case_export_with_latency() -> None:
    """The same export, plus `--exportProfile`, so every node carries its measured time.

    Without a profile the `Latency` attribute is present but empty; the attribute set does not
    depend on whether one was supplied, so a reader never has to handle two shapes of file.
    """
    run_trtexec([
        f"--onnx={onnx_file}",
        f"--exportLayerInfo={json_file.name}",
        f"--exportProfile={profile_file.name}",
        "--profilingVerbosity=detailed",
        "--iterations=10",
    ])
    export_engine_as_onnx(engine_json_file=json_file, export_onnx_file=export_file, engine_profile_file=profile_file)

    model = onnx.load(export_file)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.name == "Latency" and attribute.s:
                print(f"    {node.name}")
                print(f"        {attribute.s.decode()}")
                return
    print("    No Latency attribute carried a value")

@case_mark
def case_export_corpus() -> None:
    """A spread of engines, so the edge cases stay covered rather than being rediscovered.

    Each of these has broken `export_engine_as_onnx` at some point: `model-trained-int8-qat` carries
    Q/DQ pairs, `model-labeled` has named dimensions, and `model-redundant` collapses to a single
    layer. The check is `onnx.checker`, not merely "did not raise" - a graph only Netron will open
    is not good enough, because the point of writing ONNX is that other ONNX tooling can read it.
    """
    for model_name in ["model-trained-int8-qat", "model-labeled", "model-redundant"]:
        corpus_json_file = output_path / f"data-layer_info-{model_name}.json"
        run_trtexec([
            f"--onnx={cookbook_path('00-Data', 'model', f'{model_name}.onnx')}",
            f"--exportLayerInfo={corpus_json_file.name}",
            "--profilingVerbosity=detailed",
            "--builderOptimizationLevel=0",
            "--skipInference",
        ])
        corpus_export_file = output_path / f"result-engine-{model_name}.onnx"
        export_engine_as_onnx(engine_json_file=corpus_json_file, export_onnx_file=corpus_export_file)
        onnx.checker.check_model(onnx.load(corpus_export_file))
        print(f"    {model_name}: onnx.checker passed")

@case_mark
def case_cycle_is_viewable() -> None:
    """A hand-built ONNX with a real cycle, to justify what the exporter does with a Loop.

    This file is the evidence behind the default. `onnx.checker` rejects a cycle - ONNX is defined
    as single static assignment - but Netron draws it correctly, as a pair of arrows between the two
    nodes. Since these exports are read and never executed, a viewer that renders the cycle is worth
    more than a checker that accepts the file, so the exporter emits the cycle by default.

    It is built here rather than committed so that the claim is re-tested rather than remembered.
    """
    tensor_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    tensor_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "loop_back"], ["add_out"], name="Add_in_cycle"),
            helper.make_node("Relu", ["add_out"], ["loop_back"], name="Relu_in_cycle"),  # closes the cycle
            helper.make_node("Identity", ["loop_back"], ["y"], name="Exit"),
        ],
        "cyclic-no-subgraph",
        [tensor_x],
        [tensor_y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, cyclic_reference_file)

    onnx.load(cyclic_reference_file)  # Readable, which is all a viewer needs
    try:
        onnx.checker.check_model(model)
        raise AssertionError("Expected onnx.checker to reject a cyclic graph")
    except onnx.checker.ValidationError as e:
        print(f"    {cyclic_reference_file.name}: onnx.load works, onnx.checker rejects it as expected")
        print(f"        {str(e).splitlines()[0]}")

@case_mark
def case_control_flow() -> None:
    """An engine containing a `Loop`, exported both ways.

    TensorRT compiles a `Loop` into a flat instruction stream with a back-edge, so a tensor name has
    several producers and one layer reads and writes the same name.

    + **default** keeps those names, so the file has the cycle in it and a viewer draws the loop.
    + **`b_break_cycle=True`** renames every write to an SSA version and adds a `BackEdge` marker
      node per chain. The result is a DAG that passes `onnx.checker`, which is what ONNX tooling
      such as `onnx_outliner` needs - at the cost of being harder to read, since the back-edge
      becomes a node whose target you have to look up by name.

    `main.py`'s writers do neither: they keep one producer per tensor name and drop the rest without
    a word, losing the loop's entry edges. See README.md for the count.
    """
    for model_name, description in [("model-loop", "a Loop and nothing else"), ("model-for", "a Loop with an If in its body")]:
        control_flow_json_file = output_path / f"data-layer_info-{model_name}.json"
        run_trtexec([
            f"--onnx={cookbook_path('00-Data', 'model', f'{model_name}.onnx')}",
            f"--exportLayerInfo={control_flow_json_file.name}",
            "--profilingVerbosity=detailed",
            "--skipInference",
        ])

        cyclic_file = output_path / f"result-engine-{model_name}.onnx"
        export_engine_as_onnx(engine_json_file=control_flow_json_file, export_onnx_file=cyclic_file)
        model = onnx.load(cyclic_file)
        producer_map = OrderedDict()
        for node in model.graph.node:
            for tensor_name in node.output:
                producer_map.setdefault(tensor_name, []).append(node.name)
        n_cycle = sum(1 for producer_list in producer_map.values() if len(producer_list) > 1)
        assert n_cycle > 0, f"{model_name} contains a Loop but the default export has no cycle"
        print(f"    {model_name} ({description})")
        print(f"        default        : {len(model.graph.node):3d} nodes, {n_cycle} back-edge(s) left as cycles, names unchanged")

        dag_file = output_path / f"result-engine-{model_name}-dag.onnx"
        export_engine_as_onnx(engine_json_file=control_flow_json_file, export_onnx_file=dag_file, b_break_cycle=True)
        model = onnx.load(dag_file)
        onnx.checker.check_model(model)  # The whole point of this mode
        back_edge_list = [node for node in model.graph.node if node.op_type == "BackEdge"]
        assert len(back_edge_list) == n_cycle, f"{model_name}: {n_cycle} cycles but {len(back_edge_list)} BackEdge markers"
        print(f"        b_break_cycle=True: {len(model.graph.node):3d} nodes, {len(back_edge_list)} BackEdge marker(s), onnx.checker passed")

if __name__ == "__main__":
    case_export()
    case_export_with_latency()
    case_export_corpus()
    case_cycle_is_viewable()
    case_control_flow()

    print("\nFinish")
