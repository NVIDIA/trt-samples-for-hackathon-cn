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
import os

onnxFile0 = "model-0.onnx"
onnxFile1 = "model-1.onnx"
onnxFile2 = "model-2.onnx"
nEmbedding = 256
np.random.seed(31193)

tensor0 = gs.Variable("tensor0", np.float32, ["B", nEmbedding, "T"])
constantM1 = gs.Constant("constantM1", np.ascontiguousarray(np.array([-1], dtype=np.int64)))
constantM2 = gs.Constant("constantM2", np.ascontiguousarray(np.array([-2], dtype=np.int64)))

tensor1 = gs.Variable("tensor1", np.float32, None)
node1 = gs.Node("ReduceSum", "ReduceSum", inputs=[tensor0, constantM2], outputs=[tensor1], attrs=OrderedDict([("keepdims", 1)]))

graph = gs.Graph(nodes=[node1], inputs=[tensor0], outputs=[tensor1], opset=13)

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile0)
print("Succeeded building %s!" % (onnxFile0))

# Add a pair of Transpose nodes before and after ReduceSum node, however, they will be fused by TensorRT
graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op == "ReduceSum":
        node.inputs[1] = constantM1

        tensor2 = gs.Variable("tensor2", np.float32, None)
        node2 = gs.Node("Transpose", "Transpose-0", inputs=[node.inputs[0]], outputs=[tensor2], attrs=OrderedDict([("perm", [0, 2, 1])]))
        graph.nodes.append(node2)

        tensor3 = gs.Variable("tensor3", np.float32, None)
        node3 = gs.Node("Transpose", "Transpose-1", inputs=[tensor3], outputs=[node.outputs[0]], attrs=OrderedDict([("perm", [0, 2, 1])]))
        graph.nodes.append(node3)

        node.inputs[0] = tensor2
        node.outputs[0] = tensor3

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile1)
print("Succeeded building %s!" % (onnxFile1))

# Add a pair of Transpose nodes before and after ReduceSum node, further more, add two more Identity nodes after the Tranpose nodes in case of TenorRT's fusion
graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op == "ReduceSum":
        node.inputs[1] = constantM1

        tensor2 = gs.Variable("tensor2", np.float32, None)
        node2 = gs.Node("Transpose", "Transpose-0", inputs=[node.inputs[0]], outputs=[tensor2], attrs=OrderedDict([("perm", [0, 2, 1])]))
        graph.nodes.append(node2)

        tensor3 = gs.Variable("tensor3", np.float32, None)
        node3 = gs.Node("Identity", "Identity-0", inputs=[tensor2], outputs=[tensor3])
        graph.nodes.append(node3)

        tensor4 = gs.Variable("tensor4", np.float32, None)
        node4 = gs.Node("Transpose", "Transpose-1", inputs=[tensor4], outputs=[node.outputs[0]], attrs=OrderedDict([("perm", [0, 2, 1])]))
        graph.nodes.append(node4)

        node.inputs[0] = tensor3
        node.outputs[0] = tensor4

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile2)
print("Succeeded building %s!" % (onnxFile2))

command = "trtexec --onnx=%s --verbose --useCudaGraph --noDataTransfers --minShapes=tensor0:1x256x1024 --optShapes=tensor0:1x256x1024 --maxShapes=tensor0:1x256x1024 --shapes=tensor0:1x256x1024"
os.system(command % onnxFile0)
os.system(command % onnxFile1)
os.system(command % onnxFile2)
