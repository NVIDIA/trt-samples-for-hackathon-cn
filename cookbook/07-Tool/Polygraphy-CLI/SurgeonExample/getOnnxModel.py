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

batchName = 12  # Static Shape mode
#batchName = "B"  # Dynamic Shape mode

tensorX = gs.Variable("tensorX", np.float32, [batchName, 2, 3, 4])
constant0C1 = gs.Constant("constant0C1", np.ascontiguousarray(np.array([0, 1], dtype=np.int64)))
constant2C3 = gs.Constant("constant2C3", np.ascontiguousarray(np.array([2, 3], dtype=np.int64)))

n = 0
scopeName = "A"

tensor1, n = addNode(graph, "Shape", scopeName, n, [tensorX], None, "", np.int64, [4])

tensor2, n = addNode(graph, "ReduceProd", scopeName, n, [tensor1], OrderedDict([["axes", [0]], ["keepdims", 1]]), "", np.int64, [1])

tensor3, n = addNode(graph, "Reshape", scopeName, n, [tensorX, tensor2], None, "", np.float32, ["%s*24" % str(batchName)])

tensor4, n = addNode(graph, "Gather", scopeName, n, [tensor1, constant0C1], None, "", np.int64, [2])

tensor5, n = addNode(graph, "Gather", scopeName, n, [tensor1, constant2C3], None, "", np.int64, [2])

tensor6, n = addNode(graph, "ReduceProd", scopeName, n, [tensor5], OrderedDict([["axes", [0]], ["keepdims", 1]]), "", np.int64, [1])

tensor7, n = addNode(graph, "Concat", scopeName, n, [tensor4, tensor6], OrderedDict([["axis", 0]]), "", np.int64, [4])

tensor8, n = addNode(graph, "Reshape", scopeName, n, [tensorX, tensor7], None, "", np.float32, [batchName, 2, 12])

graph.inputs = [tensorX]
graph.outputs = [tensor3, tensor8]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "modelA.onnx")
print("Succeeded create %s" % "modelA.onnx")
