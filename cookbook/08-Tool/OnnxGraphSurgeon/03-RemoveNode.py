#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-03-01.onnx")

for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        index = node.o().inputs.index(node.outputs[0])  # find the index of output tensor of this node in the next node
        node.o().inputs[index] = node.inputs[0]  # replace the input tensor of the next node as the input tensor of this node
        node.outputs = []  # a optional step: clean the output tensor of this node, so that this node can be recognized as uselesss node

graph.cleanup().toposort()  # the Add node will be removed during graph clean
onnx.save(gs.export_onnx(graph), "model-03-02.onnx")
