#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

onnx_file = f"model-{__file__.split('/')[-1].split('.')[0]}"

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
onnx.save(gs.export_onnx(graph), onnx_file + "-01.onnx")

# replace node by edit the operator type
graph = gs.import_onnx(onnx.load(onnx_file + "-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        node.op = "Sub"
        node.name = "mySub"  # it's OK to change the name of the node or not

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-02.onnx")

# repalce node by inserting new node
graph = gs.import_onnx(onnx.load(onnx_file + "-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Add" and node.name == "myAdd":
        newNode = gs.Node("Sub", "mySub", inputs=node.inputs, outputs=node.outputs)
        graph.nodes.append(newNode)
        node.outputs = []

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-03.onnx")
