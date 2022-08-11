#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, None)
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)

constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant0], outputs=[tensor2])
node2 = gs.Node("Identity", "myIdentity1", inputs=[tensor2], outputs=[tensor3])

graph0 = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
graph0.cleanup().toposort()
onnx.save(gs.export_onnx(graph0), "model-04-01.onnx")

graph1 = graph0.copy()
for node in graph1.nodes:
    if node.op == "Add" and node.name == "myAdd":
        node.op = "Sub"  # 通过修改操作类型来替换节点
        node.name = "mySub"  # 名字该改不改都行，主要是方便区分节点以及日后查找

graph1.cleanup().toposort()
onnx.save(gs.export_onnx(graph1), "model-04-02.onnx")

graph2 = graph0.copy()
for node in graph2.nodes:
    if node.op == "Add" and node.name == "myAdd":
        newNode = gs.Node("Sub", "mySub", inputs=node.inputs, outputs=node.outputs)  # 照搬输入输出张量
        graph2.nodes.append(newNode)  # 把新节点加入计算图中
        node.outputs = []  # 将原节点的输出张量设置为空

graph2.cleanup().toposort()
onnx.save(gs.export_onnx(graph2), "model-04-03.onnx")
