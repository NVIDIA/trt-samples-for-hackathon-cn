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
"""The **legacy** ONNX exporter (`torch.onnx.export(..., dynamo=False)`).

Legacy == TorchScript tracing. It is the only exporter that can wrap an
`nn.Module` subclass into an ONNX **local function** (`FunctionProto`), via
`export_modules_as_functions`. That parameter is marked *Deprecated option* in
the PyTorch documentation and the whole legacy path raises a
`DeprecationWarning` since PyTorch 2.9, but as of PyTorch 2.13 it still works
and is the only way to get a module-level sub-graph out of PyTorch.

Cases:

1. `flat`                 - baseline, no sub-module at all.
2. `function_selected`    - `export_modules_as_functions={Block}`, one shared
                            `FunctionProto` called three times.
3. `function_all`         - `export_modules_as_functions=True`, every module
                            becomes a function, so the functions nest
                            (`Net` -> `Block` -> `Linear`) and the main graph
                            shrinks to a single node.
4. `control_flow_if`      - `torch.jit.script` + a data dependent `if`, exported
                            to an ONNX `If` with `then_branch` / `else_branch`
                            sub-graphs. This is the *other* kind of sub-graph and
                            the legacy exporter can produce it too.

See `main-dynamo.py` for what the new exporter does with the same models.
"""

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
OPSET = 17

output_path = Path(__file__).parent
result = OrderedDict()  # case name -> (n_main_node, n_function, n_sub_graph_node, output_array)
data = {}  # Filled in `main()`
data_if = {}  # Filled in `main()`, `case_control_flow_if` has one more input
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

class NetIf(nn.Module):
    """A data dependent branch, the input of an ONNX `If` node."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, x, flag):
        if bool((flag.sum() > 0).item()):
            return torch.relu(self.fc(x))
        else:
            return torch.tanh(self.fc(x))

# ================================================================ Helper functions

def export_and_catch_warning(*args, **kwargs) -> list:
    """Run `torch.onnx.export` and return the deprecation warnings it raises."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        torch.onnx.export(*args, **kwargs)
    return [str(w.message) for w in warning_list if issubclass(w.category, DeprecationWarning)]

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
    for function in model.functions:
        print(f"    Function   : domain={function.domain}, name={function.name}, node={[(n.domain, n.op_type) for n in function.node]}")
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
    """Baseline: the fully unrolled graph, no sub-module at all."""
    onnx_file = output_path / "model-legacy-01-flat.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    warning_list = export_and_catch_warning(model, (torch.from_numpy(data["x"]), ), onnx_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"])
    print(f"    DeprecationWarning x {len(warning_list)}: {warning_list[0][:80] if warning_list else ''}...")

    report("flat", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_function_selected() -> None:
    """`export_modules_as_functions={Block}`: only `Block` becomes a local function.

    Note that the three call sites share **one** `FunctionProto`. The weights are
    not baked into the function body, they are passed in as extra function inputs
    (a `FunctionProto` has no `initializer` field and is not a closure), so each
    call site can carry its own weights.
    """
    onnx_file = output_path / "model-legacy-02-function_selected.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    warning_list = export_and_catch_warning(
        model,
        (torch.from_numpy(data["x"]), ),
        onnx_file,
        dynamo=False,
        opset_version=OPSET,
        input_names=["x"],
        output_names=["y"],
        export_modules_as_functions={Block},
    )
    # The `export_modules_as_functions` parameter is documented as deprecated but
    # does *not* warn on its own, only the legacy exporter as a whole warns.
    print(f"    DeprecationWarning x {len(warning_list)}, mentioning `export_modules_as_functions`: {any('export_modules_as_functions' in w for w in warning_list)}")

    model_onnx = onnx.load(onnx_file)
    function = model_onnx.functions[0]
    print(f"    Function input  : {list(function.input)}")
    print(f"    Function output : {list(function.output)}")
    print(f"    Call site name  : {[node.name for node in model_onnx.graph.node]}")

    report("function_selected", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_function_all() -> None:
    """`export_modules_as_functions=True`: every `nn.Module` becomes a function.

    The functions then nest, `Net` calls `Block` calls `Linear`, and the main
    graph is left with a single node. This is the most compact thing Netron can
    be given, but it also hides everything behind three levels of clicking.
    """
    onnx_file = output_path / "model-legacy-03-function_all.onnx"
    model = Net(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    with warnings.catch_warnings():  # `export_modules_as_functions=True` warns about assigned tensor attributes
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (torch.from_numpy(data["x"]), ),
            onnx_file,
            dynamo=False,
            opset_version=OPSET,
            input_names=["x"],
            output_names=["y"],
            export_modules_as_functions=True,
        )

    report("function_all", onnx_file, run_onnxruntime(onnx_file, data))
    return

@case_mark
def case_control_flow_if() -> None:
    """The other kind of sub-graph: a `GraphProto` attribute of a control-flow node.

    A plain Python `if` on a tensor is *traced away* (only the taken branch is
    kept), `torch.jit.script` is needed to keep it. The result is an ONNX `If`
    node holding `then_branch` / `else_branch` sub-graphs.
    """
    onnx_file = output_path / "model-legacy-04-control_flow_if.onnx"
    model = NetIf(N_C).eval()
    model_script = torch.jit.script(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model_script,
            (torch.from_numpy(data["x"]), torch.from_numpy(data_if["flag"])),
            onnx_file,
            dynamo=False,
            opset_version=OPSET,
            input_names=["x", "flag"],
            output_names=["y"],
        )

    model_onnx = onnx.load(onnx_file)
    for node in model_onnx.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                print(f"    {node.op_type}.{attribute.name} = {[n.op_type for n in attribute.g.node]}")

    for value in [1.0, -1.0]:  # Both branches are really there, the `if` was not traced away
        feed = {"x": data["x"], "flag": np.array([value], dtype=np.float32)}
        print(f"    flag={value:+.1f} -> y[0, :4] = {run_onnxruntime(onnx_file, feed)[0, :4]}")

    report("control_flow_if", onnx_file, run_onnxruntime(onnx_file, data_if))
    return

# ================================================================ Entrance

def main() -> None:
    global data, data_if, state_dict
    data = {"x": np.random.rand(N_B, N_C).astype(np.float32) * 2 - 1}
    data_if = data | {"flag": np.array([1.0], dtype=np.float32)}
    state_dict = Net(N_C, N_BLOCK).eval().state_dict()

    case_flat()
    case_function_selected()
    case_function_all()
    case_control_flow_if()

    print("\n" + "=" * 78)
    print(f"{'Case':<24}{'MainNode':>10}{'Function':>10}{'SubNode':>10}{'MaxDiffVsFlat':>18}")
    print("-" * 78)
    baseline = result["flat"][3]
    for name, (n_node, n_function, n_sub_node, output) in result.items():
        diff = np.max(np.abs(output - baseline)) if output.shape == baseline.shape and name != "control_flow_if" else float("nan")
        print(f"{name:<24}{n_node:>10}{n_function:>10}{n_sub_node:>10}{diff:>18.3e}")
    print("=" * 78)
    print("Note: `control_flow_if` is a different model, it is not compared against `flat`.")
    print("      Folding modules into local functions is numerically exact (max diff 0).")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
