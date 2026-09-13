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
"""Demo of the ONNX outliner (milestone 1).

Folds the repeated sub-graphs of a flat ONNX into shared local functions, so the
model becomes readable in Netron. The model handed to TensorRT can stay the
original flat one, but as the report shows, TensorRT imports the outlined one
just as happily (it inlines the functions again).
"""

import json
import sys
from pathlib import Path

import onnx

from tensorrt_cookbook import OutlineConfig, case_mark, cookbook_path, outline

output_path = Path(__file__).parent

# Every input model comes from the zoo next door, which is the single place they
# are defined and the same place `test_outliner.py` takes them from. `ensure`
# exports one on first use and reuses the file afterwards.
sys.path.insert(0, str(output_path.parent / "00-ModelZoo"))
from model_zoo import ensure

def show(report: dict) -> None:
    """Print the interesting rows of a report."""
    print(f"    {report['n_node_input']:>6} nodes in the input file")
    print(f"    {report['preprocess']['n_node_after']:>6} nodes after {report['preprocess']['tool']} (P0)")
    print(f"    {report['n_node_output']:>6} nodes in the main graph + {report['n_function']} local function(s), coverage {report['coverage']:.1%}")
    for pattern in report["patterns"]:
        print(f"           {pattern['name']}: {pattern['size']:>3} nodes x {pattern['n_instance']:>2} instances, "
              f"gain {pattern['gain']:>4}, interface {pattern['n_function_input']} in / {pattern['n_function_output']} out")
    if report["rejected"]:
        print(f"    rejected candidates: {report['rejected']}")
    print(f"    verification: {report['verification']}")
    return

@case_mark
def case_transformer() -> None:
    """The main target: 6 identical encoder layers."""
    report = outline(ensure("transformer_6layer"), output_path / "model-transformer-outlined.onnx")
    show(report)
    (output_path / "report-transformer.json").write_text(json.dumps(report, indent=2))

    model = onnx.load(output_path / "model-transformer-outlined.onnx")
    print(f"    main graph now reads: {[(n.domain or '-', n.op_type) for n in model.graph.node]}")
    return

@case_mark
def case_strictness() -> None:
    """Same model at every strictness level, to show what the knob does."""
    for strictness in ["L0", "L1", "L2", "L3"]:
        report = outline(ensure("transformer_6layer"), output_path / f"model-transformer-{strictness}.onnx", OutlineConfig(strictness=strictness))
        pattern_text = ", ".join(f"{p['size']}x{p['n_instance']}" for p in report["patterns"]) or "none"
        print(f"    {strictness}: main graph {report['n_node_output']:>3} nodes, "
              f"{report['n_function']} function(s) [{pattern_text}], coverage {report['coverage']:>6.1%}")
    print("    All four agree here, which is the honest answer for this model: the 6 encoder")
    print("    layers are structurally identical down to activation dtype and rank.")
    print("    On the heterogeneous `model-large.onnx` the knob does bite, L0/L1/L2 give")
    print("    110 nodes + 3 functions while L3 gives 122 nodes + 2 functions. See README.md.")
    return

@case_mark
def case_nesting() -> None:
    """`max_level` in action, on a model that really has two levels.

    `transformer_two_stage` is `(3 layers + tanh) x 2`. Level 0 folds the layer,
    because folding one layer (gain 200) beats folding a whole stage (gain 123).
    Level 1 then sees `B B B Tanh B B B Tanh` and wraps each stage.
    """
    onnx_file = ensure("transformer_two_stage")
    for max_level in [1, 2, 3]:
        report = outline(onnx_file, output_path / f"model-two_stage-L{max_level}.onnx", OutlineConfig(max_level=max_level))
        text = ", ".join(f"L{p['level']} {p['name']}={p['size']}x{p['n_instance']}" for p in report["patterns"])
        print(f"    max_level={max_level}: main graph {report['n_node_output']:>3} nodes, {report['n_function']} function(s) [{text}]")
        for pattern in report["patterns"]:
            if pattern["level"] > 0:
                print(f"          {pattern['name']} wraps {pattern['size']} nodes = {pattern['n_original_node']} original nodes per instance")
    print("    A plain 6-layer chain gets nothing at level 1 on purpose: six identical layers have")
    print("    no natural 'two groups of three', so reporting one would imply structure that is not")
    print("    there. Such homogeneous runs are refused and counted as `rejected.homogeneous_run`.")
    return

@case_mark
def case_subgraph() -> None:
    """Mining inside `Loop` / `If` bodies, where the main graph has nothing to offer.

    `loop_body_repeats` has a data-dependent outer `for`, so TorchScript keeps it
    as a real `Loop` (a constant trip count is unrolled at export, see
    `../01-SubgraphInONNX`), and a constant-trip inner `for` that *is* unrolled,
    which is what puts four identical blocks inside the body.
    """
    onnx_file = ensure("loop_body_repeats")
    for b_subgraph in [False, True]:
        report = outline(onnx_file, output_path / f"model-loop_body-{b_subgraph}.onnx", OutlineConfig(preprocess=False, subgraph=b_subgraph))
        text = ", ".join(f"{p['name']}={p['size']}x{p['n_instance']}" for p in report["patterns"]) or "nothing"
        print(f"    subgraph={str(b_subgraph):<5}: main graph {report['n_node_output']} node(s), "
              f"{report['n_function']} function(s) [{text}], sub-graphs seen {report['subgraph']['n_subgraph']} "
              f"({report['subgraph']['n_node']} nodes)")
        print(f"                verification: {report['verification']}")

    model_new = onnx.load(output_path / "model-loop_body-True.onnx")
    for node in model_new.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                print(f"    body of {node.name} now reads: {[(n.domain or '-', n.op_type) for n in attribute.g.node]}")
    print("    The main graph is one `Loop` node, so a tool that only looks at the top level")
    print("    finds nothing at all. A FunctionProto belongs to the *model*, so the body can")
    print("    call one exactly like the main graph can.")
    return

@case_mark
def case_method() -> None:
    """`method` in action on the model where it matters most."""
    onnx_file = cookbook_path("00-Data", "model") / "model-large.onnx"
    if not onnx_file.exists():
        print(f"    skipped, {onnx_file} not found")
        return
    for method in ["serial", "auto"]:
        report = outline(onnx_file, output_path / f"model-large-{method}.onnx", OutlineConfig(method=method))
        text = ", ".join(f"{p['size']}x{p['n_instance']}" for p in report["patterns"])
        print(f"    method={method:<7}: main graph {report['n_node_output']:>3} nodes, {report['n_function']} function(s) "
              f"[{text}], coverage {report['coverage']:.1%}")
        (output_path / f"model-large-{method}.onnx").unlink(missing_ok=True)  # 1.5 GB each
    print("    The 1-D search reports the block truncated when a foreign node lands inside the run;")
    print("    lockstep growth seeded with that partial answer recovers the rest. See ../03-ParallelRepeat.")
    return

@case_mark
def case_large_model() -> None:
    """A real 1.5 GB / 6119 node model, to show it scales."""
    onnx_file = cookbook_path("00-Data", "model") / "model-large.onnx"
    if not onnx_file.exists():
        print(f"    skipped, {onnx_file} not found")
        return
    report = outline(onnx_file, output_path / "model-large-outlined.onnx", OutlineConfig(max_level=2))
    show(report)
    (output_path / "report-large.json").write_text(json.dumps(report, indent=2))
    (output_path / "model-large-outlined.onnx").unlink(missing_ok=True)  # 1.5 GB, do not keep it around
    return

if __name__ == "__main__":
    case_transformer()
    case_strictness()
    case_nesting()
    case_subgraph()
    case_method()
    case_large_model()
    print("\nFinish")
