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


def addNode(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtype=None, shape=None):
    # ONLY for the node with one output tensor!
    # graph:        The ONNX graph for edition
    # nodeType:     The type of the node to add, for example, "Concat"
    # prefix:       Optimization type, for example "RemoveLoop"
    # number:       An incremental number to prevent duplicate names
    # inputlist:    The list of input tensors for the node
    # attribution:  The attribution dictionary of the node, for example, OrderedDict([('axis',0)])
    # suffix:       Extra name for marking the tensor, for example "bTensor"
    # dtype:        The data type of the output tensor (optional)
    # shape:        The shape of the output tensor (optional)
    tensorName = prefix + "-V-" + str(number) + "-" + nodeType
    nodeName = prefix + "-N-" + str(number) + "-" + nodeType
    if attribution == None:
        attribution = OrderedDict()
    if len(suffix) > 0:
        tensorName += "-" + suffix

    tensor = gs.Variable(tensorName, dtype, shape)
    node = gs.Node(nodeType, nodeName, inputs=inputList, outputs=[tensor], attrs=attribution)
    graph.nodes.append(node)
    return tensor, number + 1

# ModelA -----------------------------------------------------------------------
graph = gs.Graph(nodes=[], inputs=[], outputs=[])

tensorX = gs.Variable("tensorX", np.float32, ["B", 1, 28, 28])
constant32x1 = gs.Constant("constant32x1", np.ascontiguousarray(np.random.rand(32, 1, 5, 5).reshape(32, 1, 5, 5).astype(np.float32) * 2 - 1))
constant32 = gs.Constant("constant32", np.ascontiguousarray(np.random.rand(32).reshape(32).astype(np.float32) * 2 - 1))
constant64x32 = gs.Constant("constant64x32", np.ascontiguousarray(np.random.rand(64, 32, 5, 5).reshape(64, 32, 5, 5).astype(np.float32) * 2 - 1))
constant64 = gs.Constant("constant64", np.ascontiguousarray(np.random.rand(64).reshape(64).astype(np.float32) * 2 - 1))
constantM1Comma3136 = gs.Constant("constantM1Comma3136", np.ascontiguousarray(np.array([-1, 7 * 7 * 64], dtype=np.int64)))
constant3136x1024 = gs.Constant("constant3136x1024", np.ascontiguousarray(np.random.rand(3136, 1024).reshape(3136, 1024).astype(np.float32) * 2 - 1))
constant1024 = gs.Constant("constant1024", np.ascontiguousarray(np.random.rand(1024).reshape(1024).astype(np.float32) * 2 - 1))
constant1024x10 = gs.Constant("constant1024x10", np.ascontiguousarray(np.random.rand(1024, 10).reshape(1024, 10).astype(np.float32) * 2 - 1))
constant10 = gs.Constant("constant10", np.ascontiguousarray(np.random.rand(10).reshape(10).astype(np.float32) * 2 - 1))

n = 0
scopeName = "A"

tensor1, n = addNode(graph, "Conv", scopeName, n, [tensorX, constant32x1, constant32], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), "", np.float32, ["B", 32, 28, 28])

tensor2, n = addNode(graph, "Relu", scopeName, n, [tensor1], None, "", np.float32, ["B", 32, 28, 28])

tensor3, n = addNode(graph, "MaxPool", scopeName, n, [tensor2], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), "", np.float32, ["B", 32, 14, 14])

tensor4, n = addNode(graph, "Conv", scopeName, n, [tensor3, constant64x32, constant64], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), "", np.float32, ["B", 64, 14, 14])

tensor5, n = addNode(graph, "Relu", scopeName, n, [tensor4], None, "", np.float32, ["B", 64, 14, 14])

tensor6, n = addNode(graph, "MaxPool", scopeName, n, [tensor5], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), "", np.float32, ["B", 64, 7, 7])

tensor7, n = addNode(graph, "Transpose", scopeName, n, [tensor6], OrderedDict([["perm", [0, 2, 3, 1]]]), "", np.float32, ["B", 7, 7, 64])

tensor8, n = addNode(graph, "Reshape", scopeName, n, [tensor7, constantM1Comma3136], None, "", np.float32, ["B", 3136])

tensor9, n = addNode(graph, "MatMul", scopeName, n, [tensor8, constant3136x1024], None, "", np.float32, ["B", 1024])

tensor10, n = addNode(graph, "Add", scopeName, n, [tensor9, constant1024], None, "", np.float32, ["B", 1024])

tensor11, n = addNode(graph, "Relu", scopeName, n, [tensor10], None, "", np.float32, ["B", 1024])

tensor12, n = addNode(graph, "MatMul", scopeName, n, [tensor11, constant1024x10], None, "", np.float32, ["B", 10])

tensor13, n = addNode(graph, "Add", scopeName, n, [tensor12, constant10], None, "", np.float32, ["B", 10])

tensor14, n = addNode(graph, "Softmax", scopeName, n, [tensor13], OrderedDict([["axis", 1]]), "", np.float32, ["B", 10])

tensor15, n = addNode(graph, "ArgMax", scopeName, n, [tensor14], OrderedDict([["axis", 1], ["keepdims", 0]]), "", np.int64, ["B"])

graph.inputs = [tensorX]
graph.outputs = [tensor13, tensor15]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "modelA.onnx")
print("Succeeded create %s" % "modelA.onnx")

# ModelB -----------------------------------------------------------------------
graph = gs.Graph(nodes=[], inputs=[], outputs=[])

tensorX = gs.Variable("tensorX", np.float32, ["B"])

n = 0
scopeName = "B"

tensor1, n = addNode(graph, "Identity", scopeName, n, [tensorX], None, "", np.float32, ["B"])

tensor2, n = addNode(graph, "UnknownNode", scopeName, n, [tensor1], None, "", np.float32, ["B"])

tensor3, n = addNode(graph, "Identity", scopeName, n, [tensor2], None, "", np.float32, ["B"])

graph.inputs = [tensorX]
graph.outputs = [tensor3]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "modelB.onnx")
print("Succeeded create %s" % "modelB.onnx")
