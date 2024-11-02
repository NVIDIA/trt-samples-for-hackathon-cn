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

import json
import re
from collections import OrderedDict
from pathlib import Path
import os
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from .utils_function import (datatype_engine_to_string, layer_type_to_class, print_array_information)
from .utils_onnx import add_node, add_node_for_trt_network

def build_mnist_network_trt(
    config: trt.IBuilderConfig,
    network: trt.INetworkDefinition,
    profile: trt.IOptimizationProfile,
    is_load_weight: bool = True,
):
    """
    Build a network TensorRT network with TensorRT API based on MNIST
    """
    if is_load_weight:
        para = np.load(Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.npz")

    shape = [-1, 1, 28, 28]
    tensor = network.add_input("x", trt.float32, shape)
    profile.set_shape(tensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
    config.add_optimization_profile(profile)

    if is_load_weight:
        w = np.ascontiguousarray(para["conv1.weight"])
        b = np.ascontiguousarray(para["conv1.bias"])
    else:
        w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(32, 1, 1).astype(np.float32))
    layer = network.add_convolution_nd(tensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
    layer.name = "Convolution1"
    layer.padding_nd = [2, 2]
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation1"
    layer = network.add_pooling_nd(layer.get_output(0), trt.PoolingType.MAX, [2, 2])
    layer.name = "Pooling1"
    layer.stride_nd = [2, 2]

    if is_load_weight:
        w = np.ascontiguousarray(para["conv2.weight"])
        b = np.ascontiguousarray(para["conv2.bias"])
    else:
        w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(64, 1, 1).astype(np.float32))
    layer = network.add_convolution_nd(layer.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
    layer.name = "Convolution2"
    layer.padding_nd = [2, 2]
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation2"
    layer = network.add_pooling_nd(layer.get_output(0), trt.PoolingType.MAX, [2, 2])
    layer.name = "Pooling2"
    layer.stride_nd = [2, 2]

    layer = network.add_shuffle(layer.get_output(0))
    layer.name = "Shuffle"
    layer.reshape_dims = (-1, 64 * 7 * 7)

    if is_load_weight:
        w = np.ascontiguousarray(para["gemm1.weight"].transpose())
        b = np.ascontiguousarray(para["gemm1.bias"].reshape(1, -1))
    else:
        w = np.ascontiguousarray(np.random.rand(64 * 7 * 7, 1024).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 1024).astype(np.float32))
    constant_layer = network.add_constant(w.shape, trt.Weights(w))
    constant_layer.name = "MatrixMultiplication1Weight"
    layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    layer.name = "MatrixMultiplication1"
    constant_layer = network.add_constant(b.shape, trt.Weights(b))
    constant_layer.name = "ConstantBias1"
    layer = network.add_elementwise(layer.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.SUM)
    layer.name = "AddBias1"
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation3"

    if is_load_weight:
        w = np.ascontiguousarray(para["gemm2.weight"].transpose())
        b = np.ascontiguousarray(para["gemm2.bias"].reshape(1, -1))
    else:
        w = np.ascontiguousarray(np.random.rand(1024, 10).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
    constant_layer = network.add_constant(w.shape, trt.Weights(w))
    constant_layer.name = "MatrixMultiplication2Weight"
    layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    layer.name = "MatrixMultiplication2"
    constant_layer = network.add_constant(b.shape, trt.Weights(b))
    constant_layer.name = "ConstantBias2"
    layer = network.add_elementwise(layer.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.SUM)
    layer.name = "AddBias2"
    layer = network.add_softmax(layer.get_output(0))
    layer.name = "Softmax"
    layer.axes = 1 << 1
    layer_topk = network.add_topk(layer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
    layer_topk.name = "TopK"

    layer.get_output(0).name = "y"
    layer_topk.get_output(1).name = "z"

    return [layer.get_output(0), layer_topk.get_output(1)]

def add_mea(network, tensor, io_shape):
    """
    Add `Matrix-Multiplication layer + Elementwise layer + Activation layer` into TensorRT network
    """
    i_shape, o_shape = io_shape
    w = np.ascontiguousarray(np.random.rand(i_shape, o_shape).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(1, o_shape).astype(np.float32))
    layer_w = network.add_constant(w.shape, trt.Weights(w))
    layer = network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_w.get_output(0), trt.MatrixOperation.NONE)
    layer_b = network.add_constant(b.shape, trt.Weights(b))
    layer = network.add_elementwise(layer.get_output(0), layer_b.get_output(0), trt.ElementWiseOperation.SUM)
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    return layer.get_output(0)

def print_network(network):
    """
    print the network for debug
    """
    print(f"{'='*64} Network input / output tensors:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"Input {i:3d}:{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"Output{i:3d}:{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}")
    print(f"{'='*64} Network layers:")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"{i:4d}->[{str(layer.type)[10:]:^18s}]->{layer.name}")
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            info = f"    In {j:2d}:"
            if tensor == None:
                info += "None"
            else:
                info += f"{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}"
                if tensor.is_network_input:
                    info += " <-(NETWORK_INPUT)"
            print(info)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            info = f"    Out{j:2d}:"
            if tensor == None:
                info += "None"
            else:
                info += f"{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}"
                if tensor.is_network_output:
                    info += " <-(NETWORK_OUTPUT)"
            print(info)

        # Print attribution of ILayer
        for key in dir(layer):
            if not (key.startswith("_") or callable(layer.__getattribute__(key))):
                print(f"    {key}:{layer.__getattribute__(key)}")
        # Print attribution of exact layer type
        layer.__class__ = layer_type_to_class(layer)
        for key in dir(layer):
            if key in dir(trt.ILayer) and key != "type":
                continue
            if key == "type" and not isinstance(layer.type, trt.LayerType):
                print(f"    type:{layer.type}")
                continue
            value = layer.__getattribute__(key)
            if isinstance(value, np.ndarray):  # for weights, we only print statistic information
                print_array_information(value, "    " + key, 0)
            else:
                print(f"    {key}: {value}")

    return

def export_network_as_onnx(network, export_onnx_file: Path = None, b_onnx_type: bool = False):
    """
    Export TensorRT network as a "ONNX-like" file, which can be opend by software like Netron
    """
    print(f"[ExportONNX]The operators in exported {export_onnx_file} might have different meaning than ONNX framework.")
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])
    graph.name = "" if network.name == "Unnamed Network 0" else network.name
    n = 0

    global_tensor_map = {}  # mapping from TRT tensor (trt.ITensor) to GS tensor (gs.Variable)
    for i in range(network.num_inputs):
        trt_tensor = network.get_input(i)
        gs_tensor = gs.Variable(trt_tensor.name, trt.nptype(trt_tensor.dtype), trt_tensor.shape)
        global_tensor_map[trt_tensor] = gs_tensor
        if gs_tensor not in graph.inputs:
            graph.inputs.append(gs_tensor)

    for i in range(network.num_layers):
        layer = network.get_layer(i)

        input_tensor_list = []
        for j in range(layer.num_inputs):
            trt_tensor = layer.get_input(j)
            if trt_tensor is None:  # Useful for constant layer
                gs_tensor = None
            elif trt_tensor in global_tensor_map.keys():  # already in the map
                gs_tensor = global_tensor_map[trt_tensor]
            else:
                print(f"[ExportONNX]Layer input tensor not in global_tensor_map: {trt_tensor.name}")  # ■
                gs_tensor = gs.Variable(trt_tensor.name, trt.nptype(trt_tensor.dtype), trt_tensor.shape)
                global_tensor_map[trt_tensor] = gs_tensor
            input_tensor_list.append(gs_tensor)

        output_name_list = []
        output_datatype_list = []
        output_shape_list = []
        for i in range(layer.num_outputs):
            trt_tensor = layer.get_output(i)
            # Don't do this check because we need this trt_tensor to overwrite the placeholder tensor in ■
            # if trt_tensor in global_tensor_map.keys():
            #     gs_tensor = global_tensor_map[trt_tensor]
            output_name_list.append(trt_tensor.name)
            output_datatype_list.append(trt.nptype(trt_tensor.dtype))
            output_shape_list.append(trt_tensor.shape)

        # Similar work we do in print_network
        attr = OrderedDict()
        # Set attribution of ILayer
        for key in dir(layer):
            if not (key.startswith("_") or callable(layer.__getattribute__(key))):
                attr[key] = str(layer.__getattribute__(key))
        # Set attribution of exact layer type
        layer.__class__ = layer_type_to_class(layer)
        for key in dir(layer):
            if key in dir(trt.ILayer) and key != "type":
                continue
            if key == "type" and not isinstance(layer.type, trt.LayerType):
                attr["algo-type"] = str(layer.type)
                continue
            value = layer.__getattribute__(key)
            if isinstance(value, np.ndarray):  # Convert all attributions into string besides weights
                ss = f"shape={value.shape}, SumAbs={np.sum(abs(value)):.5e}, Var={np.var(value):.5f}, "
                ss += f"Max={np.max(value):.5f}, Min={np.min(value):.5f}, SAD={np.sum(np.abs(np.diff(value.reshape(-1)))):.5f}, "
                ss += f"[:5]={value.reshape(-1)[:5]}, [-5:]={value.reshape(-1)[-5:]}"
                attr[key] = ss
            else:
                attr[key] = str(value)

        output_tensor_list, n = add_node_for_trt_network(graph, layer.name, attr["type"][10:], input_tensor_list, attr, \
            output_name_list, output_datatype_list, output_shape_list, n, b_onnx_type)

        if layer.num_outputs == 1:
            global_tensor_map[layer.get_output(0)] = output_tensor_list
        else:
            for i in range(layer.num_outputs):
                global_tensor_map[layer.get_output(i)] = output_tensor_list[i]

    for i in range(network.num_outputs):
        gs_tensor = global_tensor_map[network.get_output(i)]
        if gs_tensor not in graph.outputs:
            graph.outputs.append(gs_tensor)

    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, export_onnx_file, save_as_external_data=True, all_tensors_to_one_file=True, location=export_onnx_file.name + ".weight")
    print(f"Succeed saving {export_onnx_file.name}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

    return

def get_engine_tensor_info(tensor: dict = {}):
    """
    Get information of a tensor
    """
    shape = tensor["Dimensions"]
    location = tensor["Location"] if "Location" in tensor.keys() else "Unknown"
    fd = tensor["Format/Datatype"]
    fd_list = fd.split(" ")  # Define in "runtime/common/blobInfo.cpp"
    if "format" in fd_list:
        index = fd_list.index("format")
        data_type = fd_list[index - 1]
    else:
        data_type = fd_list[-1]
    data_type = datatype_engine_to_string(data_type)
    info = f"{fd}->{location}"

    return data_type, shape, info

def is_tensor_used_later(name, tensor_list, layer_list):
    """
    Whether the tensor is used in the later part of the network
    """
    # Whether this tensor is used in the same layer
    if name in [sub_tensor["Name"] for sub_tensor in tensor_list]:
        return True
    # Whether this tensor is used in the later layers
    for sub_layer in layer_list:
        # This tensor firstly appears as input tensor in the later layers, it is useful
        if name in [tensor["Name"] for tensor in sub_layer["Inputs"]]:
            return True
        # This tensor firstly appears as output tensor in the later layers, it is useless now
        if name in [tensor["Name"] for tensor in sub_layer["Outputs"]]:
            return False

def export_engine_as_onnx(engine_json, export_onnx_file: Path = None):
    """
    Export TensorRT engine as a "ONNX-like" file, which can be opend by software like Netron
    Loop structure is not supported yet
    """
    with open(engine_json, "r") as f:
        js = json.loads(f.read())

    layer_list = js["Layers"]
    io_tensor_list = js["Bindings"]

    # Preprocess to fix duplicate name problem, O(V^2)
    reg_myelin_tensor = '(__my.+)|(__tran)(\d+)'  # for example: "__myln_k_arg__bb1_24", "__tran7010"
    global_count = 0
    for i, layer in enumerate(layer_list):
        tensor_list = layer["Outputs"]
        for j, tensor in enumerate(tensor_list):  # this tensor must appear in Outputs firstly
            if len(re.findall(reg_myelin_tensor, tensor["Name"])) == 0:
                continue
            old_name = tensor["Name"]
            new_name = tensor["Name"] + "@" + str(global_count)
            global_count += 1
            js["Layers"][i]["Outputs"][j]["Name"] = new_name

            for sub_tensor in tensor_list[(j + 1):]:
                if sub_tensor["Name"] == old_name:
                    sub_tensor["Name"] = new_name
            b_finish = False
            for sub_layer in layer_list[(i + 1):]:
                if b_finish:
                    break
                tensor_list = sub_layer["Inputs"]
                for sub_tensor in tensor_list:
                    if sub_tensor["Name"] == old_name:
                        sub_tensor["Name"] = new_name
                tensor_list = sub_layer["Outputs"]
                for sub_tensor in tensor_list:
                    if sub_tensor["Name"] == old_name:
                        b_finish = True

    # Main process of building ONNX like graph
    io_tensor_list = js["Bindings"]

    graph = gs.Graph(nodes=[], inputs=[], outputs=[])
    n = 0

    global_tensor_map = {}  # mapping from Name of TRT tensor (str) to GS tensor (gs.Variable)
    global_tensor_fd_map = {}  # mapping from Name of TRT tensor (str) to format and location of the tensor (str)
    for i, layer in enumerate(layer_list):
        input_tensor_list = []
        layer_tensor_fd_map = {}
        for j, tensor in enumerate(layer["Inputs"]):
            name = tensor["Name"]  # `name` can be duplicate in TensorRT engine
            if name in global_tensor_map.keys():  # already in the map
                if is_tensor_used_later(name, layer["Inputs"][(j + 1):], layer_list[(i + 1):]):
                    gs_tensor = global_tensor_map[name]
                    layer_tensor_fd_map[name] = global_tensor_fd_map[name]
                else:
                    gs_tensor = global_tensor_map.pop(name)
                    layer_tensor_fd_map[name] = global_tensor_fd_map.pop(name)
            else:
                data_type, shape, info = get_engine_tensor_info(tensor)
                gs_tensor = gs.Variable(name, data_type, shape)
                if is_tensor_used_later(name, layer["Inputs"][(j + 1):], layer_list[(i + 1):]):
                    global_tensor_map[name] = gs_tensor
                    global_tensor_fd_map[name] = info
                layer_tensor_fd_map[name] = info

            input_tensor_list.append(gs_tensor)
            if name in io_tensor_list and gs_tensor not in graph.inputs and gs_tensor not in graph.outputs:
                graph.inputs.append(gs_tensor)

        output_datatype_list = []
        output_shape_list = []
        for tensor in layer["Outputs"]:
            name = tensor["Name"]  # tensor["Name"] can be duplicate
            if name in global_tensor_map.keys():
                gs_tensor = global_tensor_map[name]
                print("Should NOT be here")
                raise Exception
            else:
                data_type, shape, info = get_engine_tensor_info(tensor)
                output_datatype_list.append(data_type)
                output_shape_list.append(shape)
                global_tensor_fd_map[name] = info

        attr = OrderedDict()
        for key, value in layer.items():
            if key in ["LayerType", "Name", "Inputs", "Outputs"]:
                continue
            attr[key] = str(value)

        output_tensor_list, n = add_node(graph, layer["LayerType"], input_tensor_list, attr, output_datatype_list, output_shape_list, "", "", n)
        graph.nodes[-1].name = layer["Name"]

        if len(layer["Outputs"]) == 1:  # Convert single output tensor as a list
            output_tensor_list = [output_tensor_list]

        for i in range(len(layer["Outputs"])):
            name = layer["Outputs"][i]["Name"]
            gs_tensor = output_tensor_list[i]
            gs_tensor.name = name
            global_tensor_map[name] = gs_tensor
            if name in io_tensor_list and gs_tensor not in graph.outputs:
                graph.outputs.append(gs_tensor)
            layer_tensor_fd_map[name] = global_tensor_fd_map[name]

        graph.nodes[-1].attrs["TensorInfo"] = str(layer_tensor_fd_map)

    onnx_model = gs.export_onnx(graph)
    onnx.save(
        onnx_model,
        export_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=export_onnx_file.name + ".weight",
    )
    print(f"Succeed saving {export_onnx_file.name}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

    return
