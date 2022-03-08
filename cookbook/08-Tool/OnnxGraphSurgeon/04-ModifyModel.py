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
y = gs.Variable(name="Y", dtype=np.float32, shape=shape)
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
_0 = gs.Variable(name="_0")
_1 = gs.Variable(name="_1")

nodes = [
    gs.Node(op="Mul", inputs=[a, x1], outputs=[_0]),
    gs.Node(op="Add", inputs=[_0, b], outputs=[_1]),
    gs.Node(op="Add", inputs=[x0, _1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x0, x1], outputs=[y])
onnx.save(gs.export_onnx(graph), "04-ModifyModel_0.onnx")

# 读取 .onnx 并进行调整
graph = gs.import_onnx(onnx.load("04-ModifyModel_0.onnx"))

# 找到第一个 Add 节点，去掉加数 "b"
first_add = [node for node in graph.nodes if node.op == "Add"][0]
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# 将第一个 Add 节点替换为 LeakyRelu 节点
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02

# 在第一个 Add 节点后插入一个 Identity 节点
identity_out = gs.Variable("identity_out", dtype=np.float32)
identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.append(identity)

# 修改计算图输出
graph.outputs = [identity_out]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "04-ModifyModel_1.onnx")
