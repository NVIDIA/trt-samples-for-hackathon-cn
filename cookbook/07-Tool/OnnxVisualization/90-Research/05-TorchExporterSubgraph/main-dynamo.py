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
"""The **new** ONNX exporter (`torch.onnx.export(..., dynamo=True)`, the default since PyTorch 2.9).

New == `torch.export.ExportedProgram` + onnxscript. It does **not** support
module-level sub-graphs: `export_modules_as_functions` has no effect here, so the
graph Netron shows is completely flat. It does support **control-flow**
sub-graphs (`torch.cond` -> ONNX `If` with `then_branch` / `else_branch`), which
is a different thing.

The module hierarchy is not entirely lost though: it survives as node
`metadata_props` (`namespace`, `pkg.torch.onnx.class_hierarchy`,
`pkg.torch.onnx.name_scopes`). Netron does not use those to group the graph, but
a post-export outlining tool can - which is why this case is measured here.

Cases:

1. `flat`                 - the default, 12 nodes, 0 functions.
2. `function_ignored`     - `export_modules_as_functions={Block}` is accepted
                            without any warning and produces a **byte-identical**
                            model. It is silently ignored, not rejected.
3. `metadata`             - the module hierarchy survives as node metadata, and
                            asking for an opset lower than the one the exporter
                            emits natively silently strips it.
4. `control_flow_cond`    - `torch.cond` -> ONNX `If` with two sub-graphs.

See `main-legacy.py` for what the legacy exporter does with the same models.
"""

import ast
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from tensorrt_cookbook import case_mark

np.random.seed(31193)
torch.manual_seed(31193)

N_BLOCK = 3  # Number of repeated blocks
N_C = 8  # Feature width
N_B = 2  # Batch size

output_path = Path(__file__).parent
result = OrderedDict()  # case name -> (n_main_node, n_function, n_sub_graph_node, output_array)
data = {}  # Filled in `main()`
data_cond = {}  # Filled in `main()`, `case_control_flow_cond` has one more input
state_dict = {}  # Filled in `main()`

# ================================================================ PyTorch models

class Block(nn.Module):
    """The repeated sub-module: Linear -> ReLU -> Linear -> ReLU."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

class Net(nn.Module):
    """`n` blocks in a chain, each with its own weights."""

    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.block_list = nn.ModuleList([Block(c) for _ in range(n)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x

class NetCond(nn.Module):
    """A data dependent branch written with `torch.cond`, the input of an ONNX `If` node.

    A plain Python `if` here would be *specialized away* by `torch.export`, only
    the taken branch would survive. `torch.cond` is the supported way to keep it.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, x, flag):
        return torch.cond(
            flag.sum() > 0,
            lambda a: torch.relu(self.fc(a)),
            lambda a: torch.tanh(self.fc(a)),
            (x, ),
        )

# ================================================================ Helper functions

def export(model, input_tuple: tuple, **kwargs) -> onnx.ModelProto:
    """Run the new exporter quietly and return the `ModelProto`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(model, input_tuple, dynamo=True, verbose=False, **kwargs)
    return program.model_proto

def report(name: str, onnx_file: Path, output: np.ndarray) -> None:
    """Record and print one row of the comparison table."""
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)
    n_node = len(model.graph.node)
    n_function = len(model.functions)
    n_sub_graph_node = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                n_sub_graph_node += len(attribute.g.node)
    result[name] = (n_node, n_function, n_sub_graph_node, output)
    print(f"[{name}] main-graph-node={n_node}, function={n_function}, sub-graph-node={n_sub_graph_node}")
    print(f"    Main graph : {[(node.domain, node.op_type) for node in model.graph.node]}")
    return

def run_onnxruntime(onnx_file: Path, feed: dict) -> np.ndarray:
    """Run the ONNX file with onnxruntime, both as a sanity check and as the reference value."""
    option = ort.SessionOptions()
    option.intra_op_num_threads = 1  # Keep the log free of thread-affinity noise
    session = ort.InferenceSession(str(onnx_file), option, providers=["CPUExecutionProvider"])
    return session.run(None, feed)[0]

# ================================================================ Cases

@case_mark
def case_flat() -> None:
    """The default output of the new exporter: everything unrolled into the main graph."""
    onnx_file = output_path / "model-dynamo-01-flat.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    model_onnx = export(model, (torch.from_numpy(data["x"]), ), input_names=["x"], output_names=["y"])
    onnx.save(model_onnx, onnx_file)

    report("flat", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_function_ignored() -> None:
    """`export_modules_as_functions` under `dynamo=True`.

    The parameter is still in the signature of `torch.onnx.export`, the call is
    accepted, **no warning of any kind is raised**, and the resulting model is
    byte-for-byte identical to `case_flat`. That silence is the trap: a script
    ported from the legacy exporter keeps running and quietly loses its
    sub-modules.
    """
    onnx_file = output_path / "model-dynamo-02-function_ignored.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        program = torch.onnx.export(
            model,
            (torch.from_numpy(data["x"]), ),
            dynamo=True,
            verbose=False,
            input_names=["x"],
            output_names=["y"],
            export_modules_as_functions={Block},
        )
    model_onnx = program.model_proto
    onnx.save(model_onnx, onnx_file)

    print(f"    `export_modules_as_functions` still in the signature: {'export_modules_as_functions' in __import__('inspect').signature(torch.onnx.export).parameters}")
    print(f"    Warning mentioning `export_modules_as_functions`     : {any('export_modules_as_functions' in str(w.message) for w in warning_list)}")
    reference = export(model, (torch.from_numpy(data["x"]), ), input_names=["x"], output_names=["y"])
    print(f"    Byte-identical to the export without the parameter   : {model_onnx.SerializeToString() == reference.SerializeToString()}")

    report("function_ignored", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_metadata() -> None:
    """Where the module hierarchy went: node `metadata_props`.

    The graph is flat, but every node still carries `namespace` /
    `pkg.torch.onnx.class_hierarchy` / `pkg.torch.onnx.name_scopes`, which say
    exactly which module it came from. Netron does not use them to group the
    graph, but a post-export outlining tool can.

    Watch out for the opset: asking for an opset the onnxscript version converter
    cannot handle (here, anything below 18) falls back to the ONNX C API
    converter, which **drops all `metadata_props`**. `optimize` is not the
    culprit, the downgrade is.
    """
    onnx_file = output_path / "model-dynamo-03-metadata.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)

    for kwargs in [{}, {"optimize": False}, {"opset_version": 18}, {"opset_version": 17}]:
        model_onnx = export(model, (torch.from_numpy(data["x"]), ), input_names=["x"], output_names=["y"], **kwargs)
        key_list = sorted({p.key for node in model_onnx.graph.node for p in node.metadata_props})
        print(f"    {str(kwargs):<26} -> opset={model_onnx.opset_import[0].version}, node={len(model_onnx.graph.node)}, function={len(model_onnx.functions)}, metadata key={key_list}")

    # Group the flat node list back into modules using the metadata, this is what an
    # outliner would use as a hint instead of mining the topology.
    model_onnx = export(model, (torch.from_numpy(data["x"]), ), input_names=["x"], output_names=["y"])
    onnx.save(model_onnx, onnx_file)
    group = OrderedDict()
    for node in model_onnx.graph.node:
        name_scope_list = ast.literal_eval({p.key: p.value for p in node.metadata_props}["pkg.torch.onnx.name_scopes"])
        group.setdefault(name_scope_list[1] if len(name_scope_list) > 1 else "<root>", []).append(node.op_type)
    for module_name, op_type_list in group.items():
        print(f"        {module_name:<16}{op_type_list}")
    print(f"        class_hierarchy of node 0: {[p.value for p in model_onnx.graph.node[0].metadata_props if p.key == 'pkg.torch.onnx.class_hierarchy']}")

    report("metadata", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_control_flow_cond() -> None:
    """The one sub-graph the new exporter does produce: `torch.cond` -> ONNX `If`."""
    onnx_file = output_path / "model-dynamo-04-control_flow_cond.onnx"
    model = NetCond(N_C).eval()
    model_onnx = export(model, (torch.from_numpy(data["x"]), torch.from_numpy(data_cond["flag"])), input_names=["x", "flag"], output_names=["y"])
    onnx.save(model_onnx, onnx_file)

    for node in model_onnx.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                print(f"    {node.op_type}.{attribute.name} = {[n.op_type for n in attribute.g.node]}")

    for value in [1.0, -1.0]:  # Both branches are really there, the `cond` was not specialized away
        feed = {"x": data["x"], "flag": np.array([value], dtype=np.float32)}
        print(f"    flag={value:+.1f} -> y[0, :4] = {run_onnxruntime(onnx_file, feed)[0, :4]}")

    report("control_flow_cond", onnx_file, run_onnxruntime(onnx_file, data_cond))
    return

# ================================================================ Entrance

def main() -> None:
    global data, data_cond, state_dict
    data = {"x": np.random.rand(N_B, N_C).astype(np.float32) * 2 - 1}
    data_cond = data | {"flag": np.array([1.0], dtype=np.float32)}
    state_dict = Net(N_C, N_BLOCK).eval().state_dict()

    case_flat()
    case_function_ignored()
    case_metadata()
    case_control_flow_cond()

    print("\n" + "=" * 78)
    print(f"{'Case':<24}{'MainNode':>10}{'Function':>10}{'SubNode':>10}{'MaxDiffVsFlat':>18}")
    print("-" * 78)
    baseline = result["flat"][3]
    for name, (n_node, n_function, n_sub_node, output) in result.items():
        diff = np.max(np.abs(output - baseline)) if output.shape == baseline.shape and name != "control_flow_cond" else float("nan")
        print(f"{name:<24}{n_node:>10}{n_function:>10}{n_sub_node:>10}{diff:>18.3e}")
    print("=" * 78)
    print("Note: `control_flow_cond` is a different model, it is not compared against `flat`.")
    print("      The `Function` column is 0 everywhere: the new exporter never emits a local function.")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
