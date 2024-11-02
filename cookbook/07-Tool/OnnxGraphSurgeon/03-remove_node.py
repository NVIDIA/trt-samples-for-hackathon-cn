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

from pathlib import Path

onnx_file = f"model-{Path(__file__).name.split('.')[0]}"

tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, None)
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
tensor4 = gs.Variable("tensor4", np.float32, None)

node0 = gs.Node("Identity", "Node0", inputs=[tensor0], outputs=[tensor1])
node1 = gs.Node("TrashNode", "Node1", inputs=[tensor1], outputs=[tensor2])
node2 = gs.Node("Identity", "Node2", inputs=[tensor2], outputs=[tensor3])
node3 = gs.Node("Identity", "Node3", inputs=[tensor2], outputs=[tensor4])

graph = gs.Graph(nodes=[node0, node1, node2, node3], inputs=[tensor0], outputs=[tensor3, tensor4])
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-01.onnx")

graph = gs.import_onnx(onnx.load(onnx_file + "-01.onnx"))
for node in graph.nodes:
    if node.op == "TrashNode" and node.name == "Node1":
        inputTensor = node.inputs[0]
        outputTensor = node.outputs[0]
        for subNode in graph.nodes:  # search all nodes in case of the output tensor is used by multiple nodes
            if outputTensor in subNode.inputs:
                index = subNode.inputs.index(outputTensor)
                subNode.inputs[index] = inputTensor

graph.cleanup().toposort()  # the TrashNode node will be removed during graph clean
onnx.save(gs.export_onnx(graph), onnx_file + "-02.onnx")

print("Finish")
