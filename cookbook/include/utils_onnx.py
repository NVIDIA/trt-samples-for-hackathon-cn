#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
from typing import List, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
"""
# Useless resource
# Constant of scalar value
constantS0 = gs.Constant("constantS0", np.array(0, dtype=np.int64))
# Constant of 1 dimension integer value, MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!
constant0 = gs.Constant("constant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
# Constant of >1 dimension integer value
constant1c1 = gs.Constant("const1C1", np.ascontiguousarray(np.array([1, 1], dtype=np.int64)))
"""

def add_node(
    graph: gs.Graph = None,
    node_type: str = "",
    input_list: List[gs.Variable] = [],
    attribution: OrderedDict = OrderedDict(),
    datatype_list: Union[np.dtype, List[np.dtype]] = [],
    shape_list: Union[list, List[list]] = [],
    prefix: str = "",
    suffix: str = "",
    number: int = 0,
) -> Tuple[gs.Variable, int]:
    # graph:            The ONNX graph for edition
    # node_type:        The type of the node to add, for example, "Concat"
    # inputlist:        The list of input tensors for the node
    # attribution:      The attribution dictionary of the node, for example, OrderedDict([('axis',0)])
    # datatype_list:    The data type of the output tensor (optional)
    # shape_list:       The shape of the output tensor (optional)
    # prefix:           Optimization type, for example "RemoveLoop"
    # suffix:           Extra name for marking the tensor, for example "bTensor"
    # number:           An incremental number to prevent duplicate names

    if isinstance(datatype_list, list):  # Case of multi-output
        assert len(datatype_list) == len(shape_list)  # Confirm the number of output tensor
    else:  # Case of single-output
        datatype_list = [datatype_list]
        shape_list = [shape_list]

    node_name = f"N-{number}-{node_type}"
    if prefix != "":
        node_name = f"{prefix}-" + node_name
    n_output = len(datatype_list)
    output_list = []
    for i in range(n_output):
        tensor_name = f"V-{number}-{node_type}-{i}"
        if prefix != "":
            tensor_name = f"{prefix}-" + tensor_name
        if suffix != "":
            tensor_name += f"-{suffix}"
        tensor = gs.Variable(tensor_name, datatype_list[i], shape_list[i])
        output_list.append(tensor)

    node = gs.Node(node_type, node_name, inputs=input_list, outputs=output_list, attrs=attribution)
    graph.nodes.append(node)  # Update graph inside `add_node`

    if len(output_list) == 1:  # Case of single-output
        output_list = output_list[0]
    return output_list, number + 1

def convert_type_to_onnx(node_type: str = "", attribution: OrderedDict = {}):
    if node_type == "ACTIVATION":
        convert_list = {
            "RELU": "Relu",
            "SIGMOID": "Sigmoid",
            "TANH": "Tanh",
            "LEAKY_RELU": "LeakyRelu",
            "ELU": "Elu",
            "SELU": "Selu",
            "SOFTSIGN": "Softsign",
            "SOFTPLUS": "Softplus",
            "CLIP": "Clip",
            "HARD_SIGMOID": "HardSigmoid",
            "SCALED_TANH": "ScaledTanh",
            "THRESHOLDED_RELU": "ThresholdedRelu",
        }
        # No corresponding operator for GELU_ERF, GELU_TANH
        if "algo-type" in attribution.keys() and attribution["algo-type"].split(".")[-1] in convert_list.keys():
            return convert_list[attribution["algo-type"].split(".")[-1]]
        return node_type
    if node_type == "CAST":
        return "Cast"
    if node_type == "CONCATENATION":
        return "Concat"
    if node_type == "CONSTANT":
        return "Constant"
    if node_type == "CONVOLUTION":
        return "Conv"
    if node_type == "DECONVOLUTION":
        return "Deconv"
    if node_type == "ELEMENTWISE":
        convert_list = {
            "SUM": "Add",
            "PROD": "Mul",
            "MAX": "Max",
            "MIN": "Min",
            "SUB": "Sub",
            "DIV": "Div",
            "POW": "Pow",
            "AND": "And",
            "OR": "Or",
            "XOR": "Xor",
            "EQUAL": "Equal",
            "GREATER": "Greater",
            "LESS": "Less",
        }
        if "op" in attribution.keys() and attribution["op"].split(".")[-1] in convert_list.keys():
            return convert_list[attribution["op"].split(".")[-1]]
        return node_type
    if node_type == "GATHER":
        return "Gather"
    if node_type == "LOOP":
        return "Loop"
    if node_type == "MATRIX_MULTIPLY":
        return "Gemm"
    if node_type == "POOLING":
        convert_list = {"MAX": "MaxPool", "AVERAGE": "AveragePool"}
        if "algo-type" in attribution.keys() and attribution["algo-type"].split(".")[-1] in convert_list.keys():
            return convert_list[attribution["algo-type"].split(".")[-1]]
        return node_type
    if node_type == "REDUCE":
        convert_list = {
            "SUM": "ReduceSum",
            "PROD": "ReduceProd",
            "MAX": "ReduceMax",
            "MIN": "ReduceMin",
            "AVG": "ReduceMean",
        }
        if "op" in attribution.keys() and attribution["op"].split(".")[-1] in convert_list.keys():
            return convert_list[attribution["op"].split(".")[-1]]
        return node_type
    if node_type == "SELECT":
        return "Where"
    if node_type == "SHUFFLE":
        return "Reshape"
    if node_type == "SHAPE":
        return "Shape"
    if node_type == "SLICE":
        return "Slice"
    if node_type == "SOFTMAX":
        return "Softmax"
    if node_type == "TOPK":
        return "TopK"
    if node_type == "UNARY":
        convert_list = {"SQRT": "Sqrt", "NOT": "Not"}
        if "op" in attribution.keys() and attribution["op"].split(".")[-1] in convert_list.keys():
            return convert_list[attribution["op"].split(".")[-1]]
        return node_type
    return node_type

def add_node_for_trt_network(
    graph: gs.Graph = None,
    node_name: str = "",
    node_type: str = "",
    input_list: List[gs.Variable] = [],
    attribution: OrderedDict = OrderedDict(),
    name_list: Union[str, List[str]] = "",
    datatype_list: Union[np.dtype, List[np.dtype]] = [],
    shape_list: Union[list, List[list]] = [],
    number: int = 0,
    b_onnx_type: bool = False,
) -> Tuple[gs.Variable, int]:
    """
    Simplify version of function `add_node`, and we do some beautify to it.
    """

    if isinstance(name_list, list) or isinstance(datatype_list, list) or isinstance(shape_list, list):  # Case of multi-output
        assert len(name_list) == len(datatype_list)
        assert len(name_list) == len(shape_list)
    else:  # Case of single-output
        name_list = [name_list]
        datatype_list = [datatype_list]
        shape_list = [shape_list]

    n_output = len(name_list)
    output_list = []
    for i in range(n_output):
        tensor = gs.Variable(name_list[i], datatype_list[i], shape_list[i])
        output_list.append(tensor)

    if b_onnx_type:
        node_type = convert_type_to_onnx(node_type, attribution)

    node = gs.Node(node_type, node_name, inputs=input_list, outputs=output_list, attrs=attribution)
    graph.nodes.append(node)  # Update graph inside `add_node`

    if len(output_list) == 1:  # Case of single-output
        output_list = output_list[0]
    return output_list, number + 1

def mark_graph_output(graph, lNode, bMarkOutput=True, bMarkInput=False, lMarkOutput=None, lMarkInput=None, bRemoveOldOutput=True):
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

def find_son_node(tensor, condition):
    # find father / son node which meet the condition
    for subNode in tensor.outputs:
        if condition(subNode):
            return subNode

def print_graph(graph):
    n_max_son_node = 256

    print("================================================ Traverse the node")
    for index, node in enumerate(graph.nodes):
        attrs = "{" + "".join([str(key) + ':' + str(value) + ',' for key, value in node.attrs.items()]) + "}"
        print(f"Node{index:4d}: op={node.op}, name={node.name}, attrs={attrs}")
        for jndex, inputTensor in enumerate(node.inputs):
            print(f"    InTensor  {jndex}: {inputTensor}")
        for jndex, outputTensor in enumerate(node.outputs):
            print(f"    OutTensor {jndex}: {outputTensor}")

        fatherNodeList = []
        for i in range(n_max_son_node):
            try:
                newNode = node.i(i)
                fatherNodeList.append(newNode)
            except:
                break
        for jndex, newNode in enumerate(fatherNodeList):
            print(f"    FatherNode{jndex}: {newNode.name}")

        sonNodeList = []
        for i in range(n_max_son_node):
            try:
                newNode = node.o(i)
                sonNodeList.append(newNode)
            except:
                break
        for jndex, newNode in enumerate(sonNodeList):
            print(f"    SonNode   {jndex}: {newNode.name}")

    print("================================================ Traverse the tensor")
    for index, (name, tensor) in enumerate(graph.tensors().items()):
        print(f"Tensor{index:4d}: name={name}, desc={tensor}")
        for jndex, inputNode in enumerate(tensor.inputs):
            print(f"    InNode      {jndex}: {inputNode.name}")
        for jndex, outputNode in enumerate(tensor.outputs):
            print(f"    OutNode     {jndex}: {outputNode.name}")

        fatherTensorList = []
        for i in range(n_max_son_node):
            try:
                newTensor = tensor.i(i)
                fatherTensorList.append(newTensor)
            except:
                break
        for jndex, newTensor in enumerate(fatherTensorList):
            print(f"    FatherTensor{jndex}: {newTensor}")

        sonTensorList = []
        for i in range(n_max_son_node):
            try:
                newTensor = tensor.o(i)
                sonTensorList.append(newTensor)
            except:
                break
        for jndex, newTensor in enumerate(sonTensorList):
            print(f"    SonTensor   {jndex}: {newTensor}")

def build_mnist_network_onnx(export_file_name: str = None):
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    tensorX = gs.Variable("tensorX", np.float32, ["B", 1, 28, 28])
    constant32x1 = gs.Constant("constant32x1x5x5", np.ascontiguousarray(np.random.rand(32, 1, 5, 5).reshape(32, 1, 5, 5).astype(np.float32) * 2 - 1))
    constant32 = gs.Constant("constant32", np.ascontiguousarray(np.random.rand(32).reshape(32).astype(np.float32) * 2 - 1))
    constant64x32 = gs.Constant("constant64x32x5x5", np.ascontiguousarray(np.random.rand(64, 32, 5, 5).reshape(64, 32, 5, 5).astype(np.float32) * 2 - 1))
    constant64 = gs.Constant("constant64", np.ascontiguousarray(np.random.rand(64).reshape(64).astype(np.float32) * 2 - 1))
    constantM1Comma3136 = gs.Constant("constantM1Comma3136", np.ascontiguousarray(np.array([-1, 7 * 7 * 64], dtype=np.int64)))
    constant3136x1024 = gs.Constant("constant3136x1024", np.ascontiguousarray(np.random.rand(3136, 1024).reshape(3136, 1024).astype(np.float32) * 2 - 1))
    constant1024 = gs.Constant("constant1024", np.ascontiguousarray(np.random.rand(1024).reshape(1024).astype(np.float32) * 2 - 1))
    constant1024x10 = gs.Constant("constant1024x10", np.ascontiguousarray(np.random.rand(1024, 10).reshape(1024, 10).astype(np.float32) * 2 - 1))
    constant10 = gs.Constant("constant10", np.ascontiguousarray(np.random.rand(10).reshape(10).astype(np.float32) * 2 - 1))

    n = 0
    scope_name = "MNIST"

    tensor1, n = add_node(graph, "Conv", [tensorX, constant32x1, constant32], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), np.float32, ["B", 32, 28, 28], scope_name, "", n)

    tensor2, n = add_node(graph, "Relu", [tensor1], OrderedDict(), np.float32, ["B", 32, 28, 28], scope_name, "", n)

    tensor3, n = add_node(graph, "MaxPool", [tensor2], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), np.float32, ["B", 32, 14, 14], scope_name, "", n)

    tensor4, n = add_node(graph, "Conv", [tensor3, constant64x32, constant64], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), np.float32, ["B", 64, 14, 14], scope_name, "", n)

    tensor5, n = add_node(graph, "Relu", [tensor4], OrderedDict(), np.float32, ["B", 64, 14, 14], scope_name, "", n)

    tensor6, n = add_node(graph, "MaxPool", [tensor5], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), np.float32, ["B", 64, 7, 7], scope_name, "", n)

    tensor7, n = add_node(graph, "Transpose", [tensor6], OrderedDict([["perm", [0, 2, 3, 1]]]), np.float32, ["B", 7, 7, 64], scope_name, "", n)

    tensor8, n = add_node(graph, "Reshape", [tensor7, constantM1Comma3136], OrderedDict(), np.float32, ["B", 3136], scope_name, "", n)

    tensor9, n = add_node(graph, "MatMul", [tensor8, constant3136x1024], OrderedDict(), np.float32, ["B", 1024], scope_name, "", n)

    tensor10, n = add_node(graph, "Add", [tensor9, constant1024], OrderedDict(), np.float32, ["B", 1024], scope_name, "", n)

    tensor11, n = add_node(graph, "Relu", [tensor10], OrderedDict(), np.float32, ["B", 1024], scope_name, "", n)

    tensor12, n = add_node(graph, "MatMul", [tensor11, constant1024x10], OrderedDict(), np.float32, ["B", 10], scope_name, "", n)

    tensor13, n = add_node(graph, "Add", [tensor12, constant10], OrderedDict(), np.float32, ["B", 10], scope_name, "", n)

    tensor14, n = add_node(graph, "Softmax", [tensor13], OrderedDict([["axis", 1]]), np.float32, ["B", 10], scope_name, "", n)

    tensor15, n = add_node(graph, "ArgMax", [tensor14], OrderedDict([["axis", 1], ["keepdims", 0]]), np.int64, ["B"], scope_name, "", n)

    graph.inputs = [tensorX]
    graph.outputs = [tensor13, tensor15]

    graph.cleanup().toposort()

    onnx_model = gs.export_onnx(graph)
    if export_file_name:
        onnx.save(onnx_model, export_file_name)

    return onnx_model
