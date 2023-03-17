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
import onnx_graphsurgeon as gs

import onnx
from copy import deepcopy

def markGraphOutput(graph, lNode, bMarkOutput=True, bMarkInput=False, lMarkOutput=None, lMarkInput=None, bRemoveOldOutput=True):
    # graph:            The ONNX graph for edition
    # lNode:            The list of nodes we want to mark as output
    # bMarkOutput:      Whether to mark the output tensor(s) of the nodes in the lNode
    # bMarkInput:       Whether to mark the input tensor(s) of the nodes in the lNode
    # lMarkOutput:      The index of output tensor(s) of the node are marked as output, only available when len(lNode) == 1
    # lMarkInput:       The index of input tensor(s) of the node are marked as output, only available when len(lNode) == 1
    # bRemoveOldOutput: Whether to remove the original output of the network (cutting the graph to the node we want to mark to save ytime of building)

    # In most cases, using the first 4 parameters is enough, for example:
    #markGraphOutput(graph, ["/Conv"])                          # mark output tensor of the node "/Conv" as output
    #markGraphOutput(graph, ["/Conv"], False, True)             # mark input tensors of the node "/Conv" (input tensor + weight + bias) as output
    #markGraphOutput(graph, ["/TopK"], lMarkOutput=[1])         # mark the second output tensor of the node "/TopK" as output
    #markGraphOutput(graph, ["/Conv"], bRemoveOldOutput=False)  # mark output tensor of the node "/Conv" as output, and keep the original output of the network

    if bRemoveOldOutput:
        graph.outputs = []
    for node in graph.nodes:
        if node.name in lNode:
            if bMarkOutput:
                if lMarkOutput is None or len(lNode) > 1:
                    lMarkOutput = range(len(node.outputs))
                for index in lMarkOutput:
                    graph.outputs.append(node.outputs[index])
                    print("Mark node [%s] output tensor [%s]" % (node.name, node.outputs[index].name))
            if bMarkInput:
                if lMarkInput is None or len(lNode) > 1:
                    lMarkInput = range(len(node.inputs))
                for index in lMarkInput:
                    graph.outputs.append(node.inputs[index])
                    print("Mark node [%s] input  tensor [%s]" % (node.name, node.inputs[index].name))

    graph.cleanup().toposort()
    return len(lNode)

def addNode(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtype=None, shape=None):
    # ONLY for the node with one output tensor!!

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

