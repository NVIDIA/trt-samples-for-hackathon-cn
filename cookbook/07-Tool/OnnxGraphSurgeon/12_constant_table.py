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
"""Replace a computation that only depends on constants with a lookup table computed on the host.

Constant folding (`06-fold.py`) removes subgraphs whose inputs are *all* constant. It cannot touch
a subgraph that depends on a runtime tensor -- even when that tensor can only ever take values from
a small known set. A rotary position embedding is the canonical case: `cos(position * inv_freq)`
depends on `position_ids`, so the trigonometry survives folding and is recomputed every step, for
every token, forever.

If the set of possible positions is bounded (it always is -- by the maximum sequence length), the
whole thing can be evaluated once in NumPy and become a `Gather` from a constant table. That is what
the ChatGLM-6B pipeline does in `surgeon.py::adjustPositionEmbedding`; see `99-Todo/chatglm-6b.md`.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime

MAX_POSITION = 128
HALF_DIMENSION = 8
onnx_file = Path(f"model_{Path(__file__).name.split('.')[0]}")
onnx_file_0 = Path(str(onnx_file) + "_00.onnx")  # Trigonometry in the graph
onnx_file_1 = Path(str(onnx_file) + "_01.onnx")  # Precomputed table

inverse_frequency = np.ascontiguousarray((10000.0 ** (-np.arange(0, HALF_DIMENSION, dtype=np.float32) / HALF_DIMENSION)))

# ------------------------------------------------------------------------------------------------
# Before: `position_ids` is a runtime input, so Cast / Mul / Cos survive constant folding
tensor_position = gs.Variable("position_ids", np.int64, ["B", "L"])
tensor_float = gs.Variable("tensor_float", np.float32, ["B", "L"])
tensor_outer = gs.Variable("tensor_outer", np.float32, ["B", "L", HALF_DIMENSION])
tensor_cos = gs.Variable("cos", np.float32, ["B", "L", HALF_DIMENSION])
constant_frequency = gs.Constant("constant_frequency", inverse_frequency.reshape(1, 1, HALF_DIMENSION))

node_list = [
    gs.Node("Cast", "node_cast", inputs=[tensor_position], outputs=[tensor_float], attrs=OrderedDict([("to", onnx.TensorProto.FLOAT)])),
    gs.Node("Unsqueeze", "node_unsqueeze", inputs=[tensor_float, gs.Constant("axis_2", np.ascontiguousarray(np.array([2], dtype=np.int64)))], outputs=[gs.Variable("tensor_unsqueeze", np.float32, ["B", "L", 1])]),
]
tensor_unsqueeze = node_list[-1].outputs[0]
node_list += [
    gs.Node("Mul", "node_mul", inputs=[tensor_unsqueeze, constant_frequency], outputs=[tensor_outer]),
    gs.Node("Cos", "node_cos", inputs=[tensor_outer], outputs=[tensor_cos]),
]
graph = gs.Graph(nodes=node_list, inputs=[tensor_position], outputs=[tensor_cos], opset=17)  # Unsqueeze takes `axes` as an input since opset 13
onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnx_file_0)
print(f"[M] {onnx_file_0.name}: {len(graph.nodes)} nodes ({[node.op for node in graph.nodes]})")

# ------------------------------------------------------------------------------------------------
# After: evaluate the whole thing on the host for every position that can ever occur, then Gather.
# MUST use np.ascontiguousarray, or TensorRT will regard the shape of this Constant as (0) !!!
table = np.ascontiguousarray(np.cos(np.arange(MAX_POSITION, dtype=np.float32).reshape(-1, 1) * inverse_frequency.reshape(1, -1)))
constant_table = gs.Constant("constant_cos_table", table)  # [MAX_POSITION, HALF_DIMENSION]

tensor_position_1 = gs.Variable("position_ids", np.int64, ["B", "L"])
tensor_cos_1 = gs.Variable("cos", np.float32, ["B", "L", HALF_DIMENSION])
graph_1 = gs.Graph(
    nodes=[gs.Node("Gather", "node_gather", inputs=[constant_table, tensor_position_1], outputs=[tensor_cos_1], attrs=OrderedDict([("axis", 0)]))],
    inputs=[tensor_position_1],
    outputs=[tensor_cos_1],
    opset=17,
)
onnx.save(gs.export_onnx(graph_1.cleanup().toposort()), onnx_file_1)
print(f"[M] {onnx_file_1.name}: {len(graph_1.nodes)} nodes ({[node.op for node in graph_1.nodes]}), "
      f"table {table.shape} = {table.nbytes / 2**10:.1f} KiB of weight")

# ------------------------------------------------------------------------------------------------
# They must agree exactly: the table holds the same floats the graph would have computed
position_ids = np.arange(12, dtype=np.int64).reshape(2, 6) * 7  # Any positions below MAX_POSITION
output_list = []
for path in [onnx_file_0, onnx_file_1]:
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output_list.append(session.run(None, {"position_ids": position_ids})[0])
print(f"[M] max |before - after| = {np.abs(output_list[0] - output_list[1]).max():.3e}")
assert np.allclose(output_list[0], output_list[1], atol=1e-6)

print("[M] -> 4 nodes per step became 1, at the cost of a fixed table; the trade is weight for work")
print("[M] -> the ceiling is the catch: a position >= MAX_POSITION reads garbage instead of computing,")
print("[M]    so the table size is a hard limit that the original graph did not have")
print("Finish")
