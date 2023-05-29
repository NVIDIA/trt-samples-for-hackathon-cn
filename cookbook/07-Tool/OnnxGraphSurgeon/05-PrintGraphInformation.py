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

onnxFile = "./model-05-PrintGraphInformation.onnx"
nMaxAdjustNode = 256

# Create a ONNX graph with Onnx Graphsurgeon -----------------------------------
tensor0 = gs.Variable("tensor-0", np.float32, ["B", 1, 28, 28])

constant32x1 = gs.Constant("constant32x1", np.ascontiguousarray(np.random.rand(32, 1, 5, 5).reshape(32, 1, 5, 5).astype(np.float32) * 2 - 1))
constant32 = gs.Constant("constant32", np.ascontiguousarray(np.random.rand(32).reshape(32).astype(np.float32) * 2 - 1))
constant64x32 = gs.Constant("constant64x32", np.ascontiguousarray(np.random.rand(64, 32, 5, 5).reshape(64, 32, 5, 5).astype(np.float32) * 2 - 1))
constant64 = gs.Constant("constant64", np.ascontiguousarray(np.random.rand(64).reshape(64).astype(np.float32) * 2 - 1))
constantM1Comma3136 = gs.Constant("constantM1Comma3136", np.ascontiguousarray(np.array([-1, 7 * 7 * 64], dtype=np.int64)))
constant3136x1024 = gs.Constant("constant3136x1024", np.ascontiguousarray(np.random.rand(3136, 1024).reshape(3136, 1024).astype(np.float32) * 2 - 1))
constant1024 = gs.Constant("constant1024", np.ascontiguousarray(np.random.rand(1024).reshape(1024).astype(np.float32) * 2 - 1))
constant1024x10 = gs.Constant("constant1024x10", np.ascontiguousarray(np.random.rand(1024, 10).reshape(1024, 10).astype(np.float32) * 2 - 1))
constant10 = gs.Constant("constant10", np.ascontiguousarray(np.random.rand(10).reshape(10).astype(np.float32) * 2 - 1))

graphNodeList = []

tensor1 = gs.Variable("tensor-1", np.float32, None)
node1 = gs.Node("Conv", "Conv-1", inputs=[tensor0, constant32x1, constant32], outputs=[tensor1])
node1.attrs = OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]])
graphNodeList.append(node1)

tensor2 = gs.Variable("tensor-2", np.float32, None)
node2 = gs.Node("Relu", "ReLU-2", inputs=[tensor1], outputs=[tensor2])
graphNodeList.append(node2)

tensor3 = gs.Variable("tensor-3", np.float32, None)
node3 = gs.Node("MaxPool", "MaxPool-3", inputs=[tensor2], outputs=[tensor3])
node3.attrs = OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]])
graphNodeList.append(node3)

tensor4 = gs.Variable("tensor-4", np.float32, None)
node1 = gs.Node("Conv", "Conv-4", inputs=[tensor3, constant64x32, constant64], outputs=[tensor4])
node1.attrs = OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]])
graphNodeList.append(node1)

tensor5 = gs.Variable("tensor-5", np.float32, None)
node5 = gs.Node("Relu", "ReLU-5", inputs=[tensor4], outputs=[tensor5])
graphNodeList.append(node5)

tensor6 = gs.Variable("tensor-6", np.float32, None)
node6 = gs.Node("MaxPool", "MaxPool-6", inputs=[tensor5], outputs=[tensor6])
node6.attrs = OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]])
graphNodeList.append(node6)

tensor7 = gs.Variable("tensor-7", np.float32, None)
node7 = gs.Node("Transpose", "Transpose-7", inputs=[tensor6], outputs=[tensor7], attrs=OrderedDict([("perm", [0, 2, 3, 1])]))
graphNodeList.append(node7)

tensor8 = gs.Variable("tensor-8", np.float32, None)
node8 = gs.Node("Reshape", "Reshape-7", inputs=[tensor7, constantM1Comma3136], outputs=[tensor8])
graphNodeList.append(node8)

tensor9 = gs.Variable("tensor-9", np.float32, None)
node9 = gs.Node("MatMul", "MatMul-9", inputs=[tensor8, constant3136x1024], outputs=[tensor9])
graphNodeList.append(node9)

tensor10 = gs.Variable("tensor-10", np.float32, None)
node10 = gs.Node("Add", "Add-10", inputs=[tensor9, constant1024], outputs=[tensor10])
graphNodeList.append(node10)

tensor11 = gs.Variable("tensor-11", np.float32, None)
node11 = gs.Node("Relu", "ReLU-11", inputs=[tensor10], outputs=[tensor11])
graphNodeList.append(node11)

tensor12 = gs.Variable("tensor-12", np.float32, None)
node12 = gs.Node("MatMul", "MatMul-12", inputs=[tensor11, constant1024x10], outputs=[tensor12])
graphNodeList.append(node12)

tensor13 = gs.Variable("tensor-13", np.float32, None)
node13 = gs.Node("Add", "Add-13", inputs=[tensor12, constant10], outputs=[tensor13])
graphNodeList.append(node13)

tensor14 = gs.Variable("tensor-14", np.float32, None)
node14 = gs.Node("Softmax", "Softmax-14", inputs=[tensor13], outputs=[tensor14], attrs=OrderedDict([("axis", 1)]))
graphNodeList.append(node14)

tensor15 = gs.Variable("tensor-15", np.int32, None)
node15 = gs.Node("ArgMax", "ArgMax-15", inputs=[tensor14], outputs=[tensor15], attrs=OrderedDict([("axis", 1), ("keepdims", 0)]))
graphNodeList.append(node15)

graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor15])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile)

print("# Traverse the node: ----------------------------------------------------")  # traverser by nodes and print information of node, input / output tensor, name of father / son node
for index, node in enumerate(graph.nodes):
    print("Node%4d: op=%s, name=%s, attrs=%s" % (index, node.op, node.name, "".join(["{"] + [str(key) + ":" + str(value) + ", " for key, value in node.attrs.items()] + ["}"])))
    for jndex, inputTensor in enumerate(node.inputs):
        print("\tInTensor  %d: %s" % (jndex, inputTensor))
    for jndex, outputTensor in enumerate(node.outputs):
        print("\tOutTensor %d: %s" % (jndex, outputTensor))

    fatherNodeList = []
    for i in range(nMaxAdjustNode):
        try:
            newNode = node.i(i)
            fatherNodeList.append(newNode)
        except:
            break
    for jndex, newNode in enumerate(fatherNodeList):
        print("\tFatherNode%d: %s" % (jndex, newNode.name))

    sonNodeList = []
    for i in range(nMaxAdjustNode):
        try:
            newNode = node.o(i)
            sonNodeList.append(newNode)
        except:
            break
    for jndex, newNode in enumerate(sonNodeList):
        print("\tSonNode   %d: %s" % (jndex, newNode.name))

print("# Traverse the tensor: --------------------------------------------------")  # traverser by tensors and print information of tensor, name of producer / consumer node, name of father / son tensor
for index, (name, tensor) in enumerate(graph.tensors().items()):
    print("Tensor%4d: name=%s, desc=%s" % (index, name, tensor))
    for jndex, inputNode in enumerate(tensor.inputs):
        print("\tInNode      %d: %s" % (jndex, inputNode.name))
    for jndex, outputNode in enumerate(tensor.outputs):
        print("\tOutNode     %d: %s" % (jndex, outputNode.name))

    fatherTensorList = []
    for i in range(nMaxAdjustNode):
        try:
            newTensor = tensor.i(i)
            fatherTensorList.append(newTensor)
        except:
            break
    for jndex, newTensor in enumerate(fatherTensorList):
        print("\tFatherTensor%d: %s" % (jndex, newTensor))

    sonTensorList = []
    for i in range(nMaxAdjustNode):
        try:
            newTensor = tensor.o(i)
            sonTensorList.append(newTensor)
        except:
            break
    for jndex, newTensor in enumerate(sonTensorList):
        print("\tSonTensor   %d: %s" % (jndex, newTensor))
