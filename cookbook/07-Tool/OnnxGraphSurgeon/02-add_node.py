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

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnx_file = f"model-{Path(__file__).name.split('.')[0]}"

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, None)
tensor2 = gs.Variable("tensor2", np.float32, None)

node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("Identity", "myIdentity1", inputs=[tensor1], outputs=[tensor2])

graph = gs.Graph(nodes=[node0, node1], inputs=[tensor0], outputs=[tensor2])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-01.onnx")

graph = gs.import_onnx(onnx.load(onnx_file + "-01.onnx"))  # load the graph from ONNX file
for node in graph.nodes:
    if node.op == "Identity" and node.name == "myIdentity0":  # find the place we want to add node
        constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))  # construct the new variable and node
        tensor3 = gs.Variable("tensor3", np.float32, None)
        newNode = gs.Node("Add", "myAdd", inputs=[node.outputs[0], constant0], outputs=[tensor3])

        graph.nodes.append(newNode)  # REMEMBER to add the new node into the graph
        index = node.o().inputs.index(node.outputs[0])  # find the next node
        node.o().inputs[index] = tensor3  # replace the input tensor of next node as the new tensor

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-02.onnx")

print("Finish")
