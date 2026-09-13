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
"""Find *where* a reduced-precision graph goes wrong, by marking intermediate tensors as outputs.

`tensorrt_cookbook.mark_graph_output` cuts the graph down to a node you name and makes that node's
output the graph output. Repeat it node by node and a "the FP16 answer is wrong" bug turns into a
node name, without a debugger and without touching the framework the model came from.

The workflow is the one from the ChatGLM-6B pipeline in `99-Todo/chatglm-6b.md`: run the FP32 graph
as the golden reference, run the reduced-precision graph, then walk forward until the first node
whose output already disagrees. Here onnxruntime plays both parts so the example is instant; against
a real engine the "reduced precision" side is TensorRT and everything else is the same.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
from tensorrt_cookbook import mark_graph_output

np.random.seed(31193)
onnx_file = Path(f"model_{Path(__file__).name.split('.')[0]}.onnx")

# A graph that is exact in FP32 and overflows in FP16: `((x * 300) * 300) / 90000 == x`, but FP16
# tops out at 65504, so the intermediate 90000*x -- and the constant 90000 itself -- become `inf`.
# The interesting part is that nothing `inf` or `nan` reaches the output: it silently becomes 0.
tensor_x = gs.Variable("x", np.float32, ["B", 4])
tensor_0 = gs.Variable("tensor_0", np.float32, ["B", 4])
tensor_1 = gs.Variable("tensor_1", np.float32, ["B", 4])
tensor_y = gs.Variable("y", np.float32, ["B", 4])
constant_300 = gs.Constant("constant_300", np.ascontiguousarray(np.array([300.0], dtype=np.float32)))
constant_90000 = gs.Constant("constant_90000", np.ascontiguousarray(np.array([90000.0], dtype=np.float32)))

node_list = [
    gs.Node("Mul", "node_scale_up_0", inputs=[tensor_x, constant_300], outputs=[tensor_0]),
    gs.Node("Mul", "node_scale_up_1", inputs=[tensor_0, constant_300], outputs=[tensor_1]),
    gs.Node("Div", "node_scale_down", inputs=[tensor_1, constant_90000], outputs=[tensor_y]),
]
graph = gs.Graph(nodes=node_list, inputs=[tensor_x], outputs=[tensor_y])
onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnx_file)

def run(model, input_data, b_half: bool):
    """Run one model in onnxruntime, optionally with every tensor cast down to FP16."""
    if b_half:  # Emulate the reduced-precision engine: same graph, FP16 arithmetic
        model = deepcopy(model)
        for tensor in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
        for initializer in model.graph.initializer:
            array = onnx.numpy_helper.to_array(initializer).astype(np.float16)
            initializer.CopyFrom(onnx.numpy_helper.from_array(array, initializer.name))
        input_data = input_data.astype(np.float16)
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": input_data})[0].astype(np.float32)

input_data = np.arange(8, dtype=np.float32).reshape(2, 4) + 1

# The symptom: the two runs disagree, and the graph output alone does not say where
model = onnx.load(onnx_file)
output_fp32 = run(model, input_data, False)
output_fp16 = run(model, input_data, True)
print(f"[M] FP32 output: {output_fp32.reshape(-1)}")
print(f"[M] FP16 output: {output_fp16.reshape(-1)}")
print(f"[M] agree: {np.allclose(output_fp32, output_fp16)}  <- and the FP16 answer carries no inf/nan to warn you")

# The bisect: cut the graph at each node in turn, compare that node's output in both precisions.
# `b_remove_old_output=True` (the default) drops the original output, so each probe builds and runs
# only the prefix of the graph -- on a real model that is the difference between seconds and minutes.
print("[M] walking the graph forward, first disagreement wins:")
for node_name in ["node_scale_up_0", "node_scale_up_1", "node_scale_down"]:
    probe_graph = gs.import_onnx(onnx.load(onnx_file))
    mark_graph_output(probe_graph, [node_name])
    probe_model = gs.export_onnx(probe_graph)

    value_fp32 = run(probe_model, input_data, False)
    value_fp16 = run(probe_model, input_data, True)
    b_agree = np.allclose(value_fp32, value_fp16)
    print(f"[M]     {node_name:16s} FP32 {value_fp32.reshape(-1)[:2]} FP16 {value_fp16.reshape(-1)[:2]} agree={b_agree}")
    if not b_agree:
        print(f"[M] -> the error is introduced by [{node_name}], not by the node that produced the wrong final output")
        break

# `b_mark_input=True` answers the follow-up question: were the inputs of the guilty node already bad?
probe_graph = gs.import_onnx(onnx.load(onnx_file))
mark_graph_output(probe_graph, ["node_scale_up_1"], b_mark_output=False, b_mark_input=True)
print(f"[M] inputs of node_scale_up_1 marked as outputs: {[t.name for t in probe_graph.outputs]}")

print("Finish")
