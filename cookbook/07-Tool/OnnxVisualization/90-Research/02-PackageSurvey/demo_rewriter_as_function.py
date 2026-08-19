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
"""How far do the existing packages get us? -> `onnxscript.rewriter` with `as_function=True`.

`RewriteRule(..., as_function=True)` turns every match of a *user written* pattern
into a call to a shared model-local `FunctionProto`. That is exactly the second
half of what we want. What is missing is the first half: nobody discovers the
pattern for us, `target_pattern` has to be typed by hand.

This script builds the same flat 4-block MLP as `01-SubgraphInONNX`, hand-writes
the `MatMul -> Add -> Relu -> MatMul -> Add -> Relu` pattern, and shows that the
graph really collapses to 4 function-call nodes.
"""

from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort
from onnxscript import ir
from onnxscript.rewriter import RewriteRule, RewriteRuleSet

np.random.seed(31193)

N_BLOCK = 4
N_C = 8
N_B = 2
OPSET = 17

output_path = Path(__file__).parent

def build_flat_model(onnx_file: Path) -> None:
    """N unrolled `MatMul -> Add -> Relu -> MatMul -> Add -> Relu` blocks."""
    node_list, initializer_list = [], []
    tensor_name = "x"
    for i in range(N_BLOCK):
        for key, shape in [("w1", [N_C, N_C]), ("b1", [N_C]), ("w2", [N_C, N_C]), ("b2", [N_C])]:
            value = (np.random.rand(*shape).astype(np.float32) - 0.5)
            initializer_list.append(oh.make_tensor(f"{key}_{i}", onnx.TensorProto.FLOAT, shape, value.reshape(-1)))
        node_list += [
            oh.make_node("MatMul", [tensor_name, f"w1_{i}"], [f"t0_{i}"], f"MatMul1_{i}"),
            oh.make_node("Add", [f"t0_{i}", f"b1_{i}"], [f"t1_{i}"], f"Add1_{i}"),
            oh.make_node("Relu", [f"t1_{i}"], [f"t2_{i}"], f"Relu1_{i}"),
            oh.make_node("MatMul", [f"t2_{i}", f"w2_{i}"], [f"t3_{i}"], f"MatMul2_{i}"),
            oh.make_node("Add", [f"t3_{i}", f"b2_{i}"], [f"t4_{i}"], f"Add2_{i}"),
            oh.make_node("Relu", [f"t4_{i}"], [f"t5_{i}"], f"Relu2_{i}"),
        ]
        tensor_name = f"t5_{i}"
    node_list.append(oh.make_node("Identity", [tensor_name], ["y"], "OutputIdentity"))

    graph = oh.make_graph(
        node_list,
        "Flat",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_B, N_C])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_B, N_C])],
        initializer_list,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    return

# ---------------- The pattern has to be written by hand, this is the whole point
def block_pattern(op, x, w1, b1, w2, b2):
    """`target_pattern`: what to look for."""
    t0 = op.MatMul(x, w1)
    t1 = op.Add(t0, b1)
    t2 = op.Relu(t1)
    t3 = op.MatMul(t2, w2)
    t4 = op.Add(t3, b2)
    return op.Relu(t4)

def block_replacement(op, x, w1, b1, w2, b2):
    """`replacement_pattern`: what to put there instead.

    With `as_function=True` the replacement must be *exactly one* node, the
    rewriter then moves a copy of the matched nodes into a `FunctionProto`
    named after that node.
    """
    return op.MlpBlock(x, w1, b1, w2, b2, _domain="cookbook")

def main() -> None:
    flat_file = output_path / "model-flat.onnx"
    function_file = output_path / "model-function.onnx"
    build_flat_model(flat_file)

    model_flat = onnx.load(flat_file)
    print(f"Before: main-graph nodes = {len(model_flat.graph.node)}, functions = {len(model_flat.functions)}")

    rule = RewriteRule(block_pattern, block_replacement, as_function=True, name="MlpBlock")
    model_ir = ir.serde.deserialize_model(model_flat)
    n_match = RewriteRuleSet([rule]).apply_to_model(model_ir)
    model_new = ir.serde.serialize_model(model_ir)
    onnx.save(model_new, function_file)

    print(f"Matches replaced: {n_match}")
    print(f"After : main-graph nodes = {len(model_new.graph.node)}, functions = {len(model_new.functions)}")
    print(f"        main-graph node list = {[(n.domain, n.op_type) for n in model_new.graph.node]}")
    for f in model_new.functions:
        # Note the `overload` field: the rewriter emits one FunctionProto *per match*,
        # it does not merge structurally identical bodies into a single shared function.
        print(f"        function {f.domain}::{f.name}(overload={f.overload!r}) body = {[n.op_type for n in f.node]}")

    # The rewritten model must still give the same numbers
    data = {"x": (np.random.rand(N_B, N_C).astype(np.float32) - 0.5)}
    out_flat = ort.InferenceSession(str(flat_file), providers=["CPUExecutionProvider"]).run(None, data)[0]
    out_function = ort.InferenceSession(str(function_file), providers=["CPUExecutionProvider"]).run(None, data)[0]
    print(f"Max |flat - function| = {np.max(np.abs(out_flat - out_function)):.3e}")

    # And TensorRT must still be able to import it (it inlines the functions again)
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(function_file))
    print(f"TensorRT parse of the function version: ok={ok}, network layers={network.num_layers}")
    for i in range(parser.num_errors):
        print(f"    {parser.get_error(i)}")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
