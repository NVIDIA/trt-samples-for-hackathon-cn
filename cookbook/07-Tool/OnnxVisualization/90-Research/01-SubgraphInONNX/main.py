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
"""Can TensorRT recognize / build / run an ONNX file which contains sub-modules?

ONNX has two ways to express "a sub-module" instead of a flat node list:

1. Local function (`FunctionProto` in `model.functions`), a node whose `op_type`
   refers to the function and whose `domain` is a custom one. This is what
   `torch.onnx.export(..., export_modules_as_functions={...})` produces.
2. Sub-graph attribute of a control-flow node (`Loop` / `Scan` / `If`), i.e. a
   `GraphProto` living inside a node's attribute. This is what a `for` loop in
   TorchScript is exported to.

This example builds both flavours out of exactly the same PyTorch weights and
checks whether the TensorRT ONNX parser accepts them, how many TensorRT layers
each one is turned into, and whether the engine gives the same result as the
flat (fully unrolled) baseline.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort
import tensorrt as trt
import torch
import torch.nn as nn

from tensorrt_cookbook import TRTWrapperShapeInput, TRTWrapperV1, case_mark

np.random.seed(31193)
torch.manual_seed(31193)

N_BLOCK = 4  # Number of repeated blocks
N_C = 8  # Feature width
N_B = 2  # Batch size
OPSET = 17

output_path = Path(__file__).parent
result = OrderedDict()  # case name -> (n_main_node, n_function, n_sub_graph_node, n_trt_layer, output_array)
data = {}  # Filled in `main()`
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

class NetIndependentWeight(nn.Module):
    """N blocks, each with its own weights. This is the usual "node explosion" case."""

    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.block_list = nn.ModuleList([Block(c) for _ in range(n)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x

class NetSharedWeight(nn.Module):
    """One block applied n times. Weights are shared, so a real `Loop` is possible."""

    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.block = Block(c)
        self.n = n

    def forward(self, x):
        for _ in range(self.n):
            x = self.block(x)
        return x

class NetSharedWeightDynamicTrip(nn.Module):
    """Same as above but the trip count comes from an input tensor at runtime."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.block = Block(c)

    def forward(self, x, n):
        for _ in range(int(n)):
            x = self.block(x)
        return x

# ================================================================ Helper functions

def report(name: str, onnx_file: Path, output: np.ndarray, n_trt_layer: int) -> None:
    """Record and print one row of the comparison table."""
    model = onnx.load(onnx_file)
    n_node = len(model.graph.node)
    n_function = len(model.functions)
    n_sub_graph_node = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                n_sub_graph_node += len(attribute.g.node)
    result[name] = (n_node, n_function, n_sub_graph_node, n_trt_layer, output)
    n_control_flow = sum(node.op_type in ["Loop", "Scan", "If"] for node in model.graph.node)
    print(f"[{name}] main-graph-node={n_node}, function={n_function}, control-flow-node={n_control_flow}, sub-graph-node={n_sub_graph_node}, TRT-layer={n_trt_layer}")
    return

def run_onnxruntime(onnx_file: Path, data: dict) -> np.ndarray:
    """Run the ONNX file with onnxruntime as the numerical ground truth."""
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    return session.run(None, data)[0]

def build_and_run_trt(onnx_file: Path, data: dict, *, shape_input_name: str = None) -> tuple:
    """Parse the ONNX file, build an engine and run it. Return (output, n_layer)."""
    wrapper_class = TRTWrapperV1 if shape_input_name is None else TRTWrapperShapeInput
    tw = wrapper_class(logger=trt.Logger.ERROR)
    parser = trt.OnnxParser(tw.network, tw.logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed parsing {onnx_file}")

    n_layer = tw.network.num_layers  # Count layers *before* the builder optimizes the network
    for i in range(tw.network.num_inputs):
        tensor = tw.network.get_input(i)
        if tensor.is_shape_tensor:  # Trip-count style input, must be given as a shape input
            value = data[tensor.name].reshape(-1).tolist()
            tw.profile.set_shape_input(tensor.name, value, value, value)
        else:
            shape = list(data[tensor.name].shape)
            tw.profile.set_shape(tensor.name, shape, shape, shape)

    tw.build()
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)
    output_name = tw.tensor_name_list[tw.n_input]
    return tw.buffer[output_name][0].copy(), n_layer

# ================================================================ Cases

@case_mark
def case_flat() -> None:
    """Baseline: the fully unrolled graph, no sub-module at all."""
    onnx_file = output_path / "model-01-flat.onnx"
    model = NetIndependentWeight(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    torch.onnx.export(model, (torch.from_numpy(data["x"]), ), onnx_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"])

    output, n_layer = build_and_run_trt(onnx_file, data)
    report("flat", onnx_file, output, n_layer)
    return

@case_mark
def case_local_function() -> None:
    """Sub-module as ONNX local function (`FunctionProto`)."""
    onnx_file = output_path / "model-02-local_function.onnx"
    model = NetIndependentWeight(N_C, N_BLOCK).eval()
    model.load_state_dict(state_dict)
    torch.onnx.export(
        model,
        (torch.from_numpy(data["x"]), ),
        onnx_file,
        dynamo=False,
        opset_version=OPSET,
        input_names=["x"],
        output_names=["y"],
        export_modules_as_functions={Block},
    )

    model_onnx = onnx.load(onnx_file)
    print(f"    Local functions: {[(f.domain, f.name, len(f.node)) for f in model_onnx.functions]}")
    print(f"    Main graph nodes: {[(n.domain, n.op_type) for n in model_onnx.graph.node]}")

    output, n_layer = build_and_run_trt(onnx_file, data)
    report("local_function", onnx_file, output, n_layer)
    return

@case_mark
def case_loop_static_trip_count() -> None:
    """A `for` loop with a *compile-time constant* trip count and shared weights.

    Counter-intuitive result: even with `torch.jit.script`, the exporter treats
    `range(self.n)` as a constant-trip loop and **unrolls** it, so the resulting
    ONNX has no `Loop` node at all. To keep the loop you must make the trip
    count data dependent, see the next case.
    """
    onnx_file = output_path / "model-03-loop_static_trip.onnx"
    model = NetSharedWeight(N_C, N_BLOCK).eval()
    model.load_state_dict({k.replace("block_list.0.", "block."): v for k, v in state_dict.items() if k.startswith("block_list.0.")})
    model_script = torch.jit.script(model)
    torch.onnx.export(model_script, (torch.from_numpy(data["x"]), ), onnx_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"])

    output, n_layer = build_and_run_trt(onnx_file, data)
    report("loop_static_trip", onnx_file, output, n_layer)

    # This case reuses block 0 for all iterations, so it does *not* match `flat`
    reference = run_onnxruntime(onnx_file, data)
    print(f"    Max |TRT - onnxruntime| = {np.max(np.abs(output - reference)):.3e}")
    return

@case_mark
def case_loop_dynamic_trip_count() -> None:
    """`Loop` whose trip count is a runtime input, so the depth is data dependent."""
    onnx_file = output_path / "model-04-loop_dynamic_trip.onnx"
    model = NetSharedWeightDynamicTrip(N_C).eval()
    model.load_state_dict({k.replace("block_list.0.", "block."): v for k, v in state_dict.items() if k.startswith("block_list.0.")})
    model_script = torch.jit.script(model)
    torch.onnx.export(
        model_script,
        (torch.from_numpy(data["x"]), torch.tensor(N_BLOCK)),
        onnx_file,
        dynamo=False,
        opset_version=OPSET,
        input_names=["x", "n"],
        output_names=["y"],
    )

    for n in [1, 3, 5]:
        data_n = {"x": data["x"], "n": np.array(n, dtype=np.int64)}
        output, n_layer = build_and_run_trt(onnx_file, data_n, shape_input_name="n")
        reference = run_onnxruntime(onnx_file, data_n)
        print(f"    n={n}: TRT-layer={n_layer}, max |TRT - onnxruntime| = {np.max(np.abs(output - reference)):.3e}")
        if n == N_BLOCK - 1:
            report("loop_dynamic_trip", onnx_file, output, n_layer)
    return

@case_mark
def case_loop_stacked_weight() -> None:
    """The interesting one: rewrite the *unrolled, independent-weight* graph of
    `case_flat` into a single `Loop` whose body gathers per-iteration weights
    from stacked initializers. Numerically it must equal `case_flat`.

    This is exactly the transformation that sub-task 3 wants to automate, so we
    check here whether TensorRT is able to build and run the result at all.
    """
    onnx_file = output_path / "model-05-loop_stacked_weight.onnx"

    # ---- Stack the per-block weights along a new leading axis
    # torch.nn.Linear computes y = x @ W^T + b with W of shape [out, in], we
    # store W^T directly so that the body only needs a plain MatMul.
    w1 = np.stack([state_dict[f"block_list.{i}.fc1.weight"].numpy().T for i in range(N_BLOCK)]).astype(np.float32)
    b1 = np.stack([state_dict[f"block_list.{i}.fc1.bias"].numpy() for i in range(N_BLOCK)]).astype(np.float32)
    w2 = np.stack([state_dict[f"block_list.{i}.fc2.weight"].numpy().T for i in range(N_BLOCK)]).astype(np.float32)
    b2 = np.stack([state_dict[f"block_list.{i}.fc2.bias"].numpy() for i in range(N_BLOCK)]).astype(np.float32)

    initializer_list = [
        oh.make_tensor("W1", onnx.TensorProto.FLOAT, w1.shape, w1.reshape(-1)),
        oh.make_tensor("B1", onnx.TensorProto.FLOAT, b1.shape, b1.reshape(-1)),
        oh.make_tensor("W2", onnx.TensorProto.FLOAT, w2.shape, w2.reshape(-1)),
        oh.make_tensor("B2", onnx.TensorProto.FLOAT, b2.shape, b2.reshape(-1)),
        oh.make_tensor("trip_count", onnx.TensorProto.INT64, [], [N_BLOCK]),
        oh.make_tensor("cond", onnx.TensorProto.BOOL, [], [True]),
    ]

    # ---- Loop body: (iter, cond_in, x_in) -> (cond_out, x_out)
    body_node_list = [
        oh.make_node("Gather", ["W1", "iter"], ["w1_i"], "GatherW1", axis=0),
        oh.make_node("Gather", ["B1", "iter"], ["b1_i"], "GatherB1", axis=0),
        oh.make_node("Gather", ["W2", "iter"], ["w2_i"], "GatherW2", axis=0),
        oh.make_node("Gather", ["B2", "iter"], ["b2_i"], "GatherB2", axis=0),
        oh.make_node("MatMul", ["x_in", "w1_i"], ["t0"], "MatMul1"),
        oh.make_node("Add", ["t0", "b1_i"], ["t1"], "Add1"),
        oh.make_node("Relu", ["t1"], ["t2"], "Relu1"),
        oh.make_node("MatMul", ["t2", "w2_i"], ["t3"], "MatMul2"),
        oh.make_node("Add", ["t3", "b2_i"], ["t4"], "Add2"),
        oh.make_node("Relu", ["t4"], ["x_out"], "Relu2"),
        oh.make_node("Identity", ["cond_in"], ["cond_out"], "CondPassThrough"),
    ]
    body_graph = oh.make_graph(
        body_node_list,
        "LoopBody",
        [
            oh.make_tensor_value_info("iter", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []),
            oh.make_tensor_value_info("x_in", onnx.TensorProto.FLOAT, [N_B, N_C]),
        ],
        [
            oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
            oh.make_tensor_value_info("x_out", onnx.TensorProto.FLOAT, [N_B, N_C]),
        ],
    )

    graph = oh.make_graph(
        [oh.make_node("Loop", ["trip_count", "cond", "x"], ["y"], "MyLoop", body=body_graph)],
        "StackedWeightLoop",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_B, N_C])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_B, N_C])],
        initializer_list,
    )
    model_onnx = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model_onnx.ir_version = 10  # Keep it compatible with the parser shipped with TensorRT
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, onnx_file)

    output, n_layer = build_and_run_trt(onnx_file, data)
    report("loop_stacked_weight", onnx_file, output, n_layer)

    reference = run_onnxruntime(onnx_file, data)
    print(f"    Max |TRT - onnxruntime| = {np.max(np.abs(output - reference)):.3e}")
    return

# ================================================================ Entrance

def main() -> None:
    global data, state_dict
    data = {"x": np.random.rand(N_B, N_C).astype(np.float32) * 2 - 1}
    state_dict = NetIndependentWeight(N_C, N_BLOCK).eval().state_dict()

    case_flat()
    case_local_function()
    case_loop_static_trip_count()
    case_loop_dynamic_trip_count()
    case_loop_stacked_weight()

    print("\n" + "=" * 92)
    print(f"{'Case':<24}{'MainNode':>10}{'Function':>10}{'SubNode':>10}{'TRTLayer':>10}{'MaxDiffVsFlat':>18}")
    print("-" * 92)
    baseline = result["flat"][4]
    for name, (n_node, n_function, n_sub_node, n_layer, output) in result.items():
        diff = np.max(np.abs(output - baseline)) if output.shape == baseline.shape else float("nan")
        print(f"{name:<24}{n_node:>10}{n_function:>10}{n_sub_node:>10}{n_layer:>10}{diff:>18.3e}")
    print("=" * 92)
    print("Note: only `flat`, `local_function` and `loop_stacked_weight` share the same weights,")
    print("      the two `loop_*_trip*` cases reuse block 0 for every iteration on purpose.")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
