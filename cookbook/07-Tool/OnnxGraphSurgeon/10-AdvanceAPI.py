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

import os
import sys
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

np.random.seed(31193)

onnxFile0 = "model-10.onnx"
onnxFile1 = "model-10-01.onnx"
onnxFile2 = "model-10-02.onnx"

# Constant of 0 dimension
constantS0 = gs.Constant("constantS0", np.array(0, dtype=np.int64))
constantS2 = gs.Constant("constantS2", np.array(2, dtype=np.int64))
constantS3 = gs.Constant("constantS3", np.array(3, dtype=np.int64))
# Constant of 1 dimension integer value, MUST use np.ascontiguousarray, or TensorRT will regard the shape of this Constant as (0) !!!
constant0 = gs.Constant("constant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
constant1 = gs.Constant("constant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
# Constant of >1 dimension
constantWeight = gs.Constant("constantWeight", np.ascontiguousarray(np.random.rand(1 * 1 * 3 * 3).astype(np.float32).reshape([1, 1, 3, 3])))
constantBias = gs.Constant("constantBias", np.ascontiguousarray(np.random.rand(1 * 1 * 1 * 1).astype(np.float32).reshape([1, 1, 1, 1])))
constant307200x64 = gs.Constant("constant307200x256", np.ascontiguousarray(np.random.rand(307200 * 64).astype(np.float32).reshape([307200, 64])))

# Tool function
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
                    node.outputs[index].dtype = np.dtype(np.float32)
                    print("[M] Mark node [%s] output tensor [%s]" % (node.name, node.outputs[index].name))
            if bMarkInput:
                if lMarkInput is None or len(lNode) > 1:
                    lMarkInput = range(len(node.inputs))
                for index in lMarkInput:
                    graph.outputs.append(node.inputs[index])
                    print("[M] Mark node [%s] input  tensor [%s]" % (node.name, node.inputs[index].name))

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

    nodeName = prefix + "-N-" + str(number) + "-" + nodeType
    tensorName = prefix + "-V-" + str(number) + "-" + nodeType + (("-" + suffix) if len(suffix) > 0 else "")
    tensor = gs.Variable(tensorName, dtype, shape)
    node = gs.Node(nodeType, nodeName, inputs=inputList, outputs=[tensor], attrs=(OrderedDict() if attribution is None else attribution))
    graph.nodes.append(node)
    return tensor, number + 1

def addNodeMultipleOutput(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtypeList=None, shapeList=None):
    # ONLY for the node with multiple tensor!!
    # graph:        The ONNX graph for edition
    # nodeType:     The type of the node to add, for example, "Concat"
    # prefix:       Optimization type, for example "RemoveLoop"
    # number:       An incremental number to prevent duplicate names
    # inputlist:    The list of input tensors for the node
    # attribution:  The attribution dictionary of the node, for example, OrderedDict([('axis',0)])
    # suffix:       Extra name for marking the tensor, for example "bTensor"
    # dtypeList:    The list of the data type of the output tensor (optional)
    # shapeList:    The list of the shape of the output tensor (optional)

    nodeName = prefix + "-N-" + str(number) + "-" + nodeType

    assert len(dtypeList) == len(shapeList)
    outputList = []
    for i in range(len(dtypeList)):
        tensorName = prefix + "-V-" + str(number) + "-" + nodeType + "-" + str(i) + (("-" + suffix) if len(suffix) > 0 else "")
        tensor = gs.Variable(tensorName, dtypeList[i], shapeList[i])
        outputList.append(tensor)

    node = gs.Node(nodeType, nodeName, inputs=inputList, outputs=outputList, attrs=(OrderedDict() if attribution is None else attribution))
    graph.nodes.append(node)
    return outputList, number + 1

def removeAddSub(graph):
    scopeName = sys._getframe().f_code.co_name
    n = 0

    for node in graph.nodes:
        if node.op == "Add" and node.o().op == "Sub" and node.inputs[1].values == node.o().inputs[1].values:
            index = node.o().o().inputs.index(node.o().outputs[0])
            tensor, n = addNode(graph, "Identity", scopeName, n, [node.inputs[0]], None, "", np.dtype(np.float32), node.inputs[0].shape)
            node.o().o().inputs[index] = tensor
            n += 1
    return n

# Create a ONNX file as beginning
graph = gs.Graph()
scopeName = "WILI"  # whatever something
n = 0  # a counter to differentiate the names of the nodes

tensorInput = gs.Variable("tensorInput", np.dtype(np.float32), ["B", 1, 480, 640])

attribution = OrderedDict([["dilations", [1, 1]], ["kernel_shape", [3, 3]], ["pads", [1, 1, 1, 1]], ["strides", [1, 1]]])
tensor1, n = addNode(graph, "Conv", scopeName, n, [tensorInput, constantWeight], attribution, "", np.dtype(np.float32), ['B', 1, 480, 640])

tensor2, n = addNode(graph, "Add", scopeName, n, [tensor1, constantBias], None, "", np.dtype(np.float32), ['B', 1, 480, 640])

tensor3, n = addNode(graph, "Relu", scopeName, n, [tensor2], None, "", np.dtype(np.float32), ['B', 1, 480, 640])

tensor4, n = addNode(graph, "Add", scopeName, n, [tensor3, constant1], None, "", np.dtype(np.float32), ['B', 1, 480, 640])

tensor5, n = addNode(graph, "Sub", scopeName, n, [tensor4, constant1], None, "", np.dtype(np.float32), ['B', 1, 480, 640])

tensor6, n = addNode(graph, "Shape", scopeName, n, [tensorInput], None, "", np.dtype(np.int64), [4])  # value:(B,1,480, 640)

tensorBScalar, n = addNode(graph, "Gather", scopeName, n, [tensor6, constantS0], OrderedDict([('axis', 0)]), "tensorBScalar", np.dtype(np.int64), [])  # value: (B)

tensorB, n = addNode(graph, "Unsqueeze", scopeName, n, [tensorBScalar, constant0], None, "tensorB", np.dtype(np.int64), [1])  # value: (B)

tensorHScalar, n = addNode(graph, "Gather", scopeName, n, [tensor6, constantS2], OrderedDict([('axis', 0)]), "tensorHScalar", np.dtype(np.int64), [])  # value: (480)

tensorH, n = addNode(graph, "Unsqueeze", scopeName, n, [tensorHScalar, constant0], None, "tensorH", np.dtype(np.int64), [1])  # value: (480)

tensorWScalar, n = addNode(graph, "Gather", scopeName, n, [tensor6, constantS3], OrderedDict([('axis', 0)]), "tensorWScalar", np.dtype(np.int64), [])  # value: (640)

tensorW, n = addNode(graph, "Unsqueeze", scopeName, n, [tensorWScalar, constant0], None, "tensorW", np.dtype(np.int64), [1])  # value: (640)

tensorHW, n = addNode(graph, "Mul", scopeName, n, [tensorH, tensorW], None, "tensorHW", np.dtype(np.int64), [1])  # value: (480*640)

tensorBC1CHW, n = addNode(graph, "Concat", scopeName, n, [tensorB, constant1, tensorHW], OrderedDict([('axis', 0)]), "tensorBC1CHW", np.dtype(np.int64), [3])  # value: (B, 1, 480*640)

tensor7, n = addNode(graph, "Reshape", scopeName, n, [tensor5, tensorBC1CHW], None, "", np.dtype(np.float32), ["B", 1, 480 * 640])

tensor8, n = addNode(graph, "Squeeze", scopeName, n, [tensor7, constant1], None, "", np.dtype(np.float32), ["B", 480 * 640])

tensor9, n = addNode(graph, "MatMul", scopeName, n, [tensor8, constant307200x64], None, "", np.dtype(np.float32), ["B", 64])

graph.inputs = [tensorInput]
graph.outputs = [tensor9]
graph.cleanup().toposort()
graph.opset = 17  # node might not be supported by some old opset. For example, the shape inference of onnxruntime in polygraphy will fail if we use opset==11

# Save the model as ONNX file
# + "save_as_external_data" is used to seperate the weight and structure of the model, making it easier to copy if we are just interested in the structure. The saving process will fail without the switch if size of the model is larger than 2GiB.
# + If the model is small, "onnx.save(gs.export_onnx(graph), onnxFile0)" is enough
# + "all_tensors_to_one_file" is used to reduce the number of weight files
# + There must no directory prefix in "location" parameter
# + Clean the target weight files before saving, or the weight files will be appended to the old ones
os.system("rm -rf " + onnxFile0 + ".weight")
onnx.save(gs.export_onnx(graph), onnxFile0, save_as_external_data=True, all_tensors_to_one_file=True, location=onnxFile0.split('/')[-1] + ".weight")

# Load the model
# + If the size of the model is larger than 2GiB, laoding process must be divided into two steps: loading the structure firstly and then the weight
# + If the model is small, "onnxModel = onnx.load(onnxFile0)" is enough
onnxModel = onnx.load(onnxFile0, load_external_data=False)
onnx.load_external_data_for_model(onnxModel, ".")

# Do constant folding by polygraphy (and save it as visualization in this example)
# Sometimes this step should be skiped because some graph is not originally supported by polygraphy and TensorRT, so some manual graph surgery must be done before polygraphy take the model in this occasion
onnxModel = fold_constants(onnxModel, allow_onnxruntime_shape_inference=True)
onnx.save(onnxModel, onnxFile1, save_as_external_data=True, all_tensors_to_one_file=True, location=onnxFile1.split('/')[-1] + ".weight")

# Continue to do graph surgery by onnx-graphsurgeon
graph = gs.import_onnx(onnxModel)
#graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnxModel))  # This API can be used to infer the shape of each tensor if size of the model is less than 2GiB and polygraphy is not used before

# Print information of ONNX file before graph surgery
print("[M] %-16s: %5d Nodes, %5d tensors" % (onnxFile0, len(graph.nodes), len(graph.tensors().keys())))

# Do graph surgery and print how many subgraph is edited
print("[M] %4d RemoveAddSub" % removeAddSub(graph))

graph.cleanup().toposort()

# Print information of ONNX file after graph surgery
print("[M] %-16s: %5d Nodes, %5d tensors" % (onnxFile2, len(graph.nodes), len(graph.tensors().keys())))

# Print information of input / output tensor
for i, tensor in enumerate(graph.inputs):
    print("[M] Input [%2d]: %s, %s, %s" % (i, tensor.shape, tensor.dtype, tensor.name))
for i, tensor in enumerate(graph.outputs):
    print("[M] Output[%2d]: %s, %s, %s" % (i, tensor.shape, tensor.dtype, tensor.name))

# Do another constant folding by polygraphy and save it to ensure the model is supported by TensorRT
onnxModel = fold_constants(gs.export_onnx(graph), allow_onnxruntime_shape_inference=True)
onnx.save(onnxModel, onnxFile2, save_as_external_data=True, all_tensors_to_one_file=True, location=onnxFile2.split('/')[-1] + ".weight")

print("Finish graph surgery!")
