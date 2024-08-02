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

import os
import sys
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

sys.path.append("/trtcookbook/include")
from utils import add_node

onnx_file = f"model-{__file__.split('/')[-1].split('.')[0]}"
onnx_file_0 = onnx_file + "-00.onnx"
onnx_file_1 = onnx_file + "-01.onnx"
onnx_file_2 = onnx_file + "-02.onnx"

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
"""
def add_node(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtype=None, shape=None):
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
"""

# A example of surgery function
def removeAddSub(graph):
    scopeName = sys._getframe().f_code.co_name
    n = 0

    for node in graph.nodes:
        if node.op == "Add" and node.o().op == "Sub" and node.inputs[1].values == node.o().inputs[1].values:
            index = node.o().o().inputs.index(node.o().outputs[0])
            tensor, n = add_node(graph, "Identity", [node.inputs[0]], OrderedDict(), np.dtype(np.float32), node.inputs[0].shape, scopeName, "", n)
            node.o().o().inputs[index] = tensor
            n += 1
    return n

# Create a ONNX file as beginning
graph = gs.Graph()
scopeName = "WILI"  # whatever something
n = 0  # a counter to differentiate the names of the nodes

tensorInput = gs.Variable("tensorInput", np.dtype(np.float32), ["B", 1, 480, 640])

attribution = OrderedDict([["dilations", [1, 1]], ["kernel_shape", [3, 3]], ["pads", [1, 1, 1, 1]], ["strides", [1, 1]]])
tensor1, n = add_node(graph, "Conv", [tensorInput, constantWeight], attribution, np.dtype(np.float32), ['B', 1, 480, 640], scopeName, "", n)

tensor2, n = add_node(graph, "Add", [tensor1, constantBias], OrderedDict(), np.dtype(np.float32), ['B', 1, 480, 640], scopeName, "", n)

tensor3, n = add_node(graph, "Relu", [tensor2], OrderedDict(), np.dtype(np.float32), ['B', 1, 480, 640], scopeName, "", n)

tensor4, n = add_node(graph, "Add", [tensor3, constant1], OrderedDict(), np.dtype(np.float32), ['B', 1, 480, 640], scopeName, "", n)

tensor5, n = add_node(graph, "Sub", [tensor4, constant1], OrderedDict(), np.dtype(np.float32), ['B', 1, 480, 640], scopeName, "", n)

tensor6, n = add_node(graph, "Shape", [tensorInput], OrderedDict(), np.dtype(np.int64), [4], scopeName, "", n)  # value:(B, 1, 480, 640)

tensorBScalar, n = add_node(graph, "Gather", [tensor6, constantS0], OrderedDict([('axis', 0)]), np.dtype(np.int64), [], scopeName, "tensorBScalar", n)  # value: (B)

tensorB, n = add_node(graph, "Unsqueeze", [tensorBScalar, constant0], OrderedDict(), np.dtype(np.int64), [1], scopeName, "tensorB", n)  # value: (B)

tensorHScalar, n = add_node(graph, "Gather", [tensor6, constantS2], OrderedDict([('axis', 0)]), np.dtype(np.int64), [], scopeName, "tensorHScalar", n)  # value: (480)

tensorH, n = add_node(graph, "Unsqueeze", [tensorHScalar, constant0], OrderedDict(), np.dtype(np.int64), [1], scopeName, "tensorH", n)  # value: (480)

tensorWScalar, n = add_node(graph, "Gather", [tensor6, constantS3], OrderedDict([('axis', 0)]), np.dtype(np.int64), [], scopeName, "tensorWScalar", n)  # value: (640)

tensorW, n = add_node(graph, "Unsqueeze", [tensorWScalar, constant0], OrderedDict(), np.dtype(np.int64), [1], scopeName, "tensorW", n)  # value: (640)

tensorHW, n = add_node(graph, "Mul", [tensorH, tensorW], OrderedDict(), np.dtype(np.int64), [1], scopeName, "tensorHW", n)  # value: (480*640)

tensorBC1CHW, n = add_node(graph, "Concat", [tensorB, constant1, tensorHW], OrderedDict([('axis', 0)]), np.dtype(np.int64), [3], scopeName, "tensorBC1CHW", n)  # value: (B, 1, 480*640)

tensor7, n = add_node(graph, "Reshape", [tensor5, tensorBC1CHW], OrderedDict(), np.dtype(np.float32), ["B", 1, 480 * 640], scopeName, "", n)

tensor8, n = add_node(graph, "Squeeze", [tensor7, constant1], OrderedDict(), np.dtype(np.float32), ["B", 480 * 640], scopeName, "", n)

tensor9, n = add_node(graph, "MatMul", [tensor8, constant307200x64], OrderedDict(), np.dtype(np.float32), ["B", 64], scopeName, "", n)

graph.inputs = [tensorInput]
graph.outputs = [tensor9]
graph.cleanup().toposort()
graph.opset = 17  # node might not be supported by some old opset. For example, the shape inference of onnxruntime in polygraphy will fail if we use opset==11

# Save the model as ONNX file
# + "save_as_external_data" is used to separate the weight and structure of the model, making it easier to copy if we are just interested in the structure. The saving process will fail without the switch if size of the model is larger than 2GiB.
# + If the model is small, "onnx.save(gs.export_onnx(graph), onnx_file_0)" is enough
# + "all_tensors_to_one_file" is used to reduce the number of weight files
# + There must no directory prefix in "location" parameter
# + Clean the target weight files before saving, or the weight files will be appended to the old ones
os.system("rm -rf " + onnx_file_0 + ".weight")
onnx.save(gs.export_onnx(graph), onnx_file_0, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_0.split('/')[-1] + ".weight")

# Load the model
# + If the size of the model is larger than 2GiB, loading process must be divided into two steps: loading the structure firstly and then the weight
# + If the model is small, "onnxModel = onnx.load(onnx_file_0)" is enough
onnxModel = onnx.load(onnx_file_0, load_external_data=False)
onnx.load_external_data_for_model(onnxModel, ".")

# Do constant folding by polygraphy (and save it as visualization in this example)
# Sometimes this step should be skipped because some graph is not originally supported by polygraphy and TensorRT, so some manual graph surgery must be done before polygraphy take the model in this occasion
onnxModel = fold_constants(onnxModel, allow_onnxruntime_shape_inference=True)
onnx.save(onnxModel, onnx_file_1, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_1.split('/')[-1] + ".weight")

# Continue to do graph surgery by onnx-graphsurgeon
graph = gs.import_onnx(onnxModel)
#graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnxModel))  # This API can be used to infer the shape of each tensor if size of the model is less than 2GiB and polygraphy is not used before

# Print information of ONNX file before graph surgery
print(f"[M] {onnx_file_0:<16s}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

# Do graph surgery and print how many subgraph is edited
print(f"[M] {removeAddSub(graph)} RemoveAddSub")

graph.cleanup().toposort()

# Print information of ONNX file after graph surgery
print(f"[M] {onnx_file_2:<16s}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

# Print information of input / output tensor
for i, tensor in enumerate(graph.inputs):
    print(f"[M] Input [{i:2d}]: {tensor.shape}, {tensor.dtype}, {tensor.name}")
for i, tensor in enumerate(graph.outputs):
    print(f"[M] Output[{i:2d}]: {tensor.shape}, {tensor.dtype}, {tensor.name}")

# Do another constant folding by polygraphy and save it to ensure the model is supported by TensorRT
onnxModel = fold_constants(gs.export_onnx(graph), allow_onnxruntime_shape_inference=True)
onnx.save(onnxModel, onnx_file_2, save_as_external_data=True, all_tensors_to_one_file=True, location=onnx_file_2.split('/')[-1] + ".weight")
