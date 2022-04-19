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

from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable(name="tensor0", dtype=np.float32, shape=['B', 3, 64, 64])
tensor1 = gs.Variable(name="tensor1", dtype=np.float32, shape=None)
tensor2 = gs.Variable(name="tensor2", dtype=np.float32, shape=None)
tensor3 = gs.Variable(name="tensor3", dtype=np.float32, shape=None)

constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node(name="myIdentity0", op="Identity", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node(name="myAdd", op="Add", inputs=[tensor1, constant0], outputs=[tensor2])
node2 = gs.Node(name="myIdentity1", op="Identity", inputs=[tensor2], outputs=[tensor3])

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-03-01.onnx")

for node in graph.nodes:
    if node.op == 'Add' and node.name == 'myAdd':
        index = node.o().inputs.index(node.outputs[0])  # 小心地找到下一个节点中该张量的位置
        node.o().inputs[index] = node.inputs[0]  # 把下一节点的对应输入张量赋为 Add 节点的输入张量
        node.outputs = []  # 关键操作：将 Add 节点的输出张量设置为空，这样 Add 节点就成为无用节点，可以被自动清理删掉

graph.cleanup().toposort()  # 在清理时会自动删掉 Add 节点+
onnx.save(gs.export_onnx(graph), "model-03-02.onnx")
