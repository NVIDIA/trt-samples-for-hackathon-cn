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
x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 224, 224))
y = gs.Variable(name="y", dtype=np.float32)
_0 = gs.Variable(name="_0")
_1 = gs.Variable(name="_1")

nodes = [
    gs.Node(op="Identity", inputs=[x], outputs=[_0]),
    gs.Node(op="FakeNodeToRemove", inputs=[_0], outputs=[_1]),
    gs.Node(op="Identity", inputs=[_1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "06-RemoveNode_0.onnx")

# 读取 .onnx 并进行调整
graph = gs.import_onnx(onnx.load("06-RemoveNode_0.onnx"))

# 找到欲删除的节点
fake_node = [node for node in graph.nodes if node.op == "FakeNodeToRemove"][0]

# 获取欲删除节点的输入节点（母节点）
inp_node = fake_node.i()

# 将母节点的输出张量赋为欲删除节点的输出张量
inp_node.outputs = fake_node.outputs
fake_node.outputs.clear()

graph.cleanup()
onnx.save(gs.export_onnx(graph), "06-RemoveNode_1.onnx")
