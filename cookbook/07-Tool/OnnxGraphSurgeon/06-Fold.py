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

import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])  # 3 necessary tensors
tensor1 = gs.Variable("tensor1", np.float32, ["B", 3, 64, 64])
tensor2 = gs.Variable("tensor2", np.float32, ["B", 3, 64, 64])
tensor3 = gs.Variable("tensor3", np.float32, ["B", 3, 64, 64])  # 1 fake input tensor
tensor4 = gs.Variable("tensor4", np.float32, ["B", 1, 64, 64])  # 1 fake output tensor
tensor5 = gs.Variable("tensor5", np.float32, ["B", 1, 64, 64])  # 2 useless tensors
tensor6 = gs.Variable("tensor6", np.float32, ["B", 1, 64, 64])
tensor7 = gs.Variable("tensor7", np.float32, None)  # 2 intermediate tensors
tensor8 = gs.Variable("tensor8", np.float32, None)

constant0 = gs.Constant(name="w", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node("Add", "myAdd0", inputs=[constant0, constant0], outputs=[tensor7])
node1 = gs.Node("Add", "myAdd1", inputs=[tensor7, constant0], outputs=[tensor8])
node2 = gs.Node("Add", "myAdd2", inputs=[tensor0, tensor8], outputs=[tensor1])  # necessary node
node3 = gs.Node("Add", "myAdd3", inputs=[tensor1, constant0], outputs=[tensor2])  # necessary node
node4 = gs.Node("Add", "myAdd4", inputs=[tensor5, constant0], outputs=[tensor6])  # useless node

graph = gs.Graph(nodes=[node4, node3, node2, node1, node0], inputs=[tensor0, tensor3], outputs=[tensor2, tensor4])

onnx.save(gs.export_onnx(graph), "model-06-01.onnx")
# original graph, containing 4 tensors without node, 1 node without edge and 1 chain subgraph with constant expresssion

onnx.save(gs.export_onnx(graph.fold_constants()), "model-06-02.onnx")
# graph after constant folding, the subgraph with constant expression is merged into the main branch, but the two more Add node are left without edge
# notice that constant folding will not remove any nodes

onnx.save(gs.export_onnx(graph.fold_constants().cleanup()), "model-06-03.onnx")
# graph after clean, the 3 Add nodes without edge are removed

print("Before toposort:")  # The order of the original graph
for index, node in enumerate(graph.nodes):
    print("No.%d->%s" % (index, node.name))

print("After toposort:")  # The order of the last graph
graph.cleanup().toposort()
for index, node in enumerate(graph.nodes):
    print("No.%d->%s" % (index, node.name))

graph.inputs = [tensor0]
graph.outputs = [tensor2]  # remove redundant output manually
onnx.save(gs.export_onnx(graph), "model-06-04.onnx")
# In TensorRT<7.0, redundant input / output tensors will be removed, so the count of input / output tensor of a TensorRT engine may be different from the count in ONNX file
# In TensorRT>=8.0, redundant input / output tensors will be retained and be a useless placeholder in the network
