#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 生成 .onnx
shape = (1, 3, 224, 224)
x0 = gs.Variable(name="x0", dtype=np.float32, shape=shape)
x1 = gs.Variable(name="x1", dtype=np.float32, shape=shape)
y = gs.Variable(name="y", dtype=np.float32, shape=shape)
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
_0 = gs.Variable(name="_0")
_1 = gs.Variable(name="_1")

nodeList = [
    gs.Node(op="Mul", inputs=[a, x1], outputs=[_0]),
    gs.Node(op="Add", inputs=[_0, b], outputs=[_1]),
    gs.Node(op="Add", inputs=[x0, _1], outputs=[y]),
]

graph = gs.Graph(nodes=nodeList, inputs=[x0, x1], outputs=[y])
onnx.save(gs.export_onnx(graph), "03-IsolateSubgraph_0.onnx")

# 读取 .onnx 并进行调整
model = onnx.load("03-IsolateSubgraph_0.onnx")
#model = onnx.shape_inference.infer_shapes(onnx.load("model.onnx"))  # 带有形状推理的计算图
graph = gs.import_onnx(model)
tensors = graph.tensors()

graph.inputs = [tensors["x1"].to_variable(dtype=np.float32)]
graph.outputs = [tensors["_1"].to_variable(dtype=np.float32)]

graph.cleanup()
onnx.save(gs.export_onnx(graph), "03-IsolateSubgraph_1.onnx")
