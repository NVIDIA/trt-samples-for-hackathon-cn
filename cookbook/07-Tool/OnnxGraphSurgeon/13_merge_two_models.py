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
"""Splice a second ONNX file onto the first, then move post-processing into the graph.

Exporters often hand you a model in pieces -- a body and a head, a backbone and a detector -- and
running them as two engines means a round trip and two sets of bindings. Both halves are just
graphs, so ONNX-GraphSurgeon can join them:

+ **lift a weight out of the second file** and append it as a `gs.Constant` (what
  `surgeon.py::addTail` does in the ChatGLM-6B pipeline: the `lm_head` matrix is exported on its own
  and then folded into the transformer graph), or
+ **transplant the nodes** wholesale, which is what you need when the second file is more than one
  operator.

The example does both, then adds the piece that actually pays: an `ArgMax` tail, so the merged graph
returns a class index instead of a vector of scores. See `99-Todo/chatglm-6b.md`.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime

np.random.seed(31193)
N_FEATURE, N_CLASS = 16, 10
onnx_file = Path(f"model_{Path(__file__).name.split('.')[0]}")
body_file = Path(str(onnx_file) + "_body.onnx")
head_file = Path(str(onnx_file) + "_head.onnx")
merged_file = Path(str(onnx_file) + "_merged.onnx")

# The body: x -> Relu -> feature
tensor_x = gs.Variable("x", np.float32, ["B", N_FEATURE])
tensor_feature = gs.Variable("feature", np.float32, ["B", N_FEATURE])
body_graph = gs.Graph(
    nodes=[gs.Node("Relu", "node_body_relu", inputs=[tensor_x], outputs=[tensor_feature])],
    inputs=[tensor_x],
    outputs=[tensor_feature],
    opset=17,
)
onnx.save(gs.export_onnx(body_graph), body_file)

# The head, exported separately: feature -> MatMul(W) -> Add(b) -> score
weight = np.ascontiguousarray(np.random.rand(N_FEATURE, N_CLASS).astype(np.float32))
bias = np.ascontiguousarray(np.random.rand(N_CLASS).astype(np.float32))
tensor_head_in = gs.Variable("feature", np.float32, ["B", N_FEATURE])
tensor_matmul = gs.Variable("tensor_matmul", np.float32, ["B", N_CLASS])
tensor_score = gs.Variable("score", np.float32, ["B", N_CLASS])
head_graph = gs.Graph(
    nodes=[
        gs.Node("MatMul", "node_head_matmul", inputs=[tensor_head_in, gs.Constant("head_weight", weight)], outputs=[tensor_matmul]),
        gs.Node("Add", "node_head_bias", inputs=[tensor_matmul, gs.Constant("head_bias", bias)], outputs=[tensor_score]),
    ],
    inputs=[tensor_head_in],
    outputs=[tensor_score],
    opset=17,
)
onnx.save(gs.export_onnx(head_graph), head_file)
print(f"[M] body {len(body_graph.nodes)} nodes, head {len(head_graph.nodes)} nodes")

# ------------------------------------------------------------------------------------------------
# Merge. `load_external_data=False` keeps this cheap on a real model -- the weights are only pulled
# in for the tensors actually copied over.
graph = gs.import_onnx(onnx.load(body_file))
head = gs.import_onnx(onnx.load(head_file))

# (a) lifting a weight out of the other file: reach into its nodes and deepcopy the array. The
#     ChatGLM pipeline does exactly this for `lm_head`, because it only needed the matrix.
head_weight = gs.Constant("merged_head_weight", np.ascontiguousarray(head.nodes[0].inputs[1].values.copy()))
print(f"[M] lifted weight {head_weight.name} {head_weight.values.shape} out of {head_file.name}")

# (b) transplanting nodes: re-point the head's input at the body's output and append the nodes.
#     Tensors are shared by object, so rewiring is assignment, not surgery.
head.nodes[0].inputs[0] = graph.outputs[0]
head.nodes[0].inputs[1] = head_weight
graph.nodes += head.nodes
graph.outputs = head.outputs

# (c) the tail worth adding: the host was going to argmax anyway
tensor_class = gs.Variable("class_id", np.int64, ["B"])
graph.nodes.append(gs.Node("ArgMax", "node_argmax", inputs=[graph.outputs[0]], outputs=[tensor_class], attrs=OrderedDict([("axis", 1), ("keepdims", 0)])))
graph.outputs = [tensor_class]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), merged_file)
print(f"[M] merged: {len(graph.nodes)} nodes ({[node.op for node in graph.nodes]}), "
      f"output {graph.outputs[0].name} {graph.outputs[0].shape}")

# ------------------------------------------------------------------------------------------------
# Two sessions and one round trip vs one session
input_data = (np.random.rand(4, N_FEATURE).astype(np.float32) - 0.5) * 2
feature = onnxruntime.InferenceSession(str(body_file), providers=["CPUExecutionProvider"]).run(None, {"x": input_data})[0]
score = onnxruntime.InferenceSession(str(head_file), providers=["CPUExecutionProvider"]).run(None, {"feature": feature})[0]
reference = np.argmax(score, axis=1)

merged = onnxruntime.InferenceSession(str(merged_file), providers=["CPUExecutionProvider"]).run(None, {"x": input_data})[0]
print(f"[M] body + head + host argmax: {reference}")
print(f"[M] merged graph            : {merged}")
assert np.array_equal(reference, merged)
print(f"[M] -> one graph, and the host receives {N_CLASS} -> 1 values per sample")

print("Finish")
