# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from polygraphy.backend.onnx.loader import fold_constants

from .utils_cookbook import cookbook_path
from .utils_function import print_array_information
from .utils_onnx import add_node_v2

def build_mnist_network_trt(
    tw=None,
    builder_config: trt.IBuilderConfig | None = None,
    network: trt.INetworkDefinition | None = None,
    profile: trt.IOptimizationProfile | None = None,
    is_load_weight: bool = True,
    rng: np.random.Generator | None = None,
) -> list[trt.ITensor]:
    """
    Build a TensorRT network with TensorRT API based on MNIST
    For internal unit tests since hard-code path is used.
    """
    if tw is not None:
        builder_config = tw.builder_config
        network = tw.network
        profile = tw.profile
    else:
        assert not (builder_config is None or network is None or profile is None), "Either provide a TRTWrapperV1 or provide builder_config/network/profile separately."

    if is_load_weight:
        para = np.load(cookbook_path("00-Data", "model", "model-trained.npz"))
    else:
        rng = rng or np.random.default_rng()

    shape = [-1, 1, 28, 28]
    tensor = network.add_input("x", trt.float32, shape)
    # Configure the profile but do NOT add it to the builder config, matching
    # `load_mnist_network_trt` below. Adding it here as well as in `TRTWrapperV1.build` put the same
    # profile into the engine twice, and `engine.num_optimization_profiles` came back as 2.
    profile.set_shape(tensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])

    if is_load_weight:
        w = np.ascontiguousarray(para["conv1.weight"])
        b = np.ascontiguousarray(para["conv1.bias"])
    else:
        w = np.ascontiguousarray(rng.random((32, 1, 5, 5), dtype=np.float32))
        b = np.ascontiguousarray(rng.random((32, 1, 1), dtype=np.float32))
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
        w = np.ascontiguousarray(rng.random((64, 32, 5, 5), dtype=np.float32))
        b = np.ascontiguousarray(rng.random((64, 1, 1), dtype=np.float32))
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
        w = np.ascontiguousarray(rng.random((64 * 7 * 7, 1024), dtype=np.float32))
        b = np.ascontiguousarray(rng.random((1, 1024), dtype=np.float32))
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
        w = np.ascontiguousarray(rng.random((1024, 10), dtype=np.float32))
        b = np.ascontiguousarray(rng.random((1, 10), dtype=np.float32))
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

def load_large_network_trt(
    tw=None,
    logger: trt.Logger | None = None,
    builder_config: trt.IBuilderConfig | None = None,
    network: trt.INetworkDefinition | None = None,
    profile: trt.IOptimizationProfile | None = None,
):
    """
    Build a TensorRT network with ONNX parser based on wenet
    For internal unit tests since hard-code path is used.
    """
    if tw is not None:
        logger = tw.logger
        builder_config = tw.builder_config
        network = tw.network
        profile = tw.profile
    else:
        assert not (logger is None or builder_config is None or network is None or profile is None), "Either provide a TRTWrapperV1 or provide logger/builder_config/network/profile separately."

    onnx_model = onnx.load(cookbook_path("00-Data", "model", "model-large.onnx"))
    onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_onnx_path = Path(tmp_dir) / "model-large-poly.onnx"
        onnx.save(
            onnx_model,
            temp_onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=temp_onnx_path.name + ".weight",
        )

        parse_onnx(temp_onnx_path, logger, network, builder_config)

    profile.set_shape("input_ids", [1, 4], [2, 32], [4, 128])
    profile.set_shape("attention_mask", [1, 4], [2, 32], [4, 128])
    builder_config.add_optimization_profile(profile)

    return []

def load_mnist_network_trt(
    tw=None,
    logger: trt.Logger | None = None,
    builder_config: trt.IBuilderConfig | None = None,
    network: trt.INetworkDefinition | None = None,
    profile: trt.IOptimizationProfile | None = None,
    b_dynamic_shape: bool = True,
):
    """Load and parse the MNIST ONNX model, then attach an optimization profile."""
    if tw is not None:
        logger = tw.logger
        builder_config = tw.builder_config
        network = tw.network
        profile = tw.profile
    else:
        assert not (logger is None and builder_config is None and network is None and profile is None), "Either provide a TRTWrapperV1 or provide logger/builder_config/network/profile separately."

    onnx_model_path = cookbook_path("00-Data", "model", "model-trained.onnx")
    parse_onnx(onnx_model_path, logger, network, builder_config)

    if b_dynamic_shape:
        profile.set_shape("x", [1, 1, 28, 28], [2, 1, 28, 28], [4, 1, 28, 28])
    else:
        profile.set_shape("x", [1, 1, 28, 28], [1, 1, 28, 28], [1, 1, 28, 28])
    builder_config.add_optimization_profile(profile)

    return

def add_mea(network, tensor, io_shape, rng: np.random.Generator = None):
    """
    Add `Matrix-Multiplication layer + Elementwise layer + Activation layer` into TensorRT network
    """
    i_shape, o_shape = io_shape
    rng = rng or np.random.default_rng()
    w = np.ascontiguousarray(rng.random((i_shape, o_shape), dtype=np.float32))
    b = np.ascontiguousarray(rng.random((1, o_shape), dtype=np.float32))
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
        print(f"{i:4d}->[{layer_type_to_layer_type_name(layer.type):^18s}]->{layer.name}")
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            info = f"    In {j:2d}:"
            if tensor is None:
                info += "None"
            else:
                info += f"{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}"
                if tensor.is_network_input:
                    for i in range(network.num_inputs):
                        if network.get_input(i) == tensor:
                            info += f" <-[NETWORK_INPUT({i})]"
                            break
            print(info)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            info = f"    Out{j:2d}:"
            if tensor is None:
                info += "None"
            else:
                info += f"{tensor.shape},{str(tensor.dtype)[9:]},{str(tensor.location)[15:]},{tensor.name}"
                if tensor.is_network_output:
                    for i in range(network.num_outputs):
                        if network.get_output(i) == tensor:
                            info += f" <-[NETWORK_OUTPUT({i})]"
                if network.is_debug_tensor(tensor):
                    info += f"<-[NETWORK_DEBUG_TENSOR]"
            print(info)

        # Print attribution of ILayer
        for key in dir(layer):
            if not (key.startswith("_") or callable(layer.__getattribute__(key))):
                print(f"    {key}:{layer.__getattribute__(key)}")
        # Print attribution of exact layer type
        layer_dynamic_cast(layer)
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

    placeholder_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)

        input_tensor_list = []
        for j in range(layer.num_inputs):
            trt_tensor = layer.get_input(j)
            if trt_tensor is None:  # Useful for constant layer or certain None input
                placeholder_name = f"PlaceHolder_{placeholder_count}"
                placeholder_count += 1
                gs_tensor = gs.Variable(placeholder_name, np.uint64, [])
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
        for j in range(layer.num_outputs):
            trt_tensor = layer.get_output(j)
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
        layer_dynamic_cast(layer)
        for key in dir(layer):
            if key in dir(trt.ILayer) and key != "type":
                continue
            if key == "type" and not isinstance(layer.type, trt.LayerType):
                attr["algo-type"] = str(layer.type)
                continue
            value = layer.__getattribute__(key)
            if isinstance(value, np.ndarray):  # Convert all attributions into string besides weights
                if value.size == 0:  # Empty array
                    value = np.array(-np.finfo(np.float32).max, dtype=np.float32)
                ss = f"shape={value.shape}, SumAbs={np.sum(abs(value)):.5e}, Var={np.var(value):.5f}, "
                ss += f"Max={np.max(value):.5f}, Min={np.min(value):.5f}, SAD={np.sum(np.abs(np.diff(value.reshape(-1)))):.5f}, "
                ss += f"[:5]={value.reshape(-1)[:5]}, [-5:]={value.reshape(-1)[-5:]}"
                attr[key] = ss
            else:
                attr[key] = str(value)

        output_tensor_list, n = add_node_v2(graph, layer.name, attr["type"][10:], input_tensor_list, attr, \
            output_name_list, output_datatype_list, output_shape_list, n, b_onnx_type)

        if layer.num_outputs == 1:
            global_tensor_map[layer.get_output(0)] = output_tensor_list
        else:
            for j in range(layer.num_outputs):
                global_tensor_map[layer.get_output(j)] = output_tensor_list[j]

    for i in range(network.num_outputs):
        gs_tensor = global_tensor_map[network.get_output(i)]
        if gs_tensor not in graph.outputs:
            graph.outputs.append(gs_tensor)

    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, export_onnx_file, save_as_external_data=True, all_tensors_to_one_file=True, location=export_onnx_file.name + ".weight")
    print(f"Succeed saving {export_onnx_file.name}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

    return

########################################################################################################################
# Layer type helpers

def print_layer_class():
    """
    Layer name map in TensorRT-10.16:
    [print(f"{int(value):2d}", type_name, layer_name) for (type_name, (value, layer_name)) in trt.LayerType.__entries.items()]
    | Layer Type Value |  Layer Type Name   |         Layer Name          |   Add Layer Method Name    |
    | :--------------: | :----------------: | :-------------------------: | :------------------------: |
    |        0         |    CONVOLUTION     |      IConvolutionLayer      |     add_convolution_nd     |
    |        1         |        CAST        |         ICastLayer          |          add_cast          |
    |        2         |     ACTIVATION     |      IActivationLayer       |       add_activation       |
    |        3         |      POOLING       |        IPoolingLayer        |       add_pooling_nd       |
    |        4         |        LRN         |          ILRNLayer          |          add_lrn           |
    |        5         |       SCALE        |         IScaleLayer         |  add_scale / add_scale_nd  |
    |        6         |      SOFTMAX       |        ISoftMaxLayer        |        add_softmax         |
    |        7         |   DECONVOLUTION    |     IDeconvolutionLayer     |    add_deconvolution_nd    |
    |        8         |   CONCATENATION    |     IConcatenationLayer     |     add_concatenation      |
    |        9         |    ELEMENTWISE     |      IElementWiseLayer      |      add_elementwise       |
    |        10        |       PLUGIN       |              /              |         add_plugin         |
    |        11        |       UNARY        |         IUnaryLayer         |         add_unary          |
    |        12        |      PADDING       |        IPaddingLayer        |       add_padding_nd       |
    |        13        |      SHUFFLE       |        IShuffleLayer        |        add_shuffle         |
    |        14        |       REDUCE       |        IReduceLayer         |         add_reduce         |
    |        15        |        TOPK        |         ITopKLayer          |          add_topk          |
    |        16        |       GATHER       |        IGatherLayer         | add_gather / add_gather_v2 |
    |        17        |  MATRIX_MULTIPLY   |    IMatrixMultiplyLayer     |    add_matrix_multiply     |
    |        18        |   RAGGED_SOFTMAX   |     IRaggedSoftMaxLayer     |     add_ragged_softmax     |
    |        19        |      CONSTANT      |       IConstantLayer        |        add_constant        |
    |        20        |      IDENTITY      |       IIdentityLayer        |        add_identity        |
    |        21        |     PLUGIN_V2      |       IPluginV2Layer        |       add_plugin_v2        |
    |        22        |       SLICE        |         ISliceLayer         |         add_slice          |
    |        23        |       SHAPE        |         IShapeLayer         |         add_shape          |
    |        24        |  PARAMETRIC_RELU   |    IParametricReLULayer     |    add_parametric_relu     |
    |        25        |       RESIZE       |        IResizeLayer         |         add_resize         |
    |        26        |     TRIP_LIMIT     |       ITripLimitLayer       |       add_trip_limit       |
    |        27        |     RECURRENCE     |      IRecurrenceLayer       |             /              |
    |        28        |      ITERATOR      |       IIteratorLayer        |             /              |
    |        29        |    LOOP_OUTPUT     |      ILoopOutputLayer       |             /              |
    |        30        |       SELECT       |        ISelectLayer         |         add_select         |
    |        31        |        FILL        |         IFillLayer          |          add_fill          |
    |        32        |      QUANTIZE      |       IQuantizeLayer        |        add_quantize        |
    |        33        |     DEQUANTIZE     |      IDequantizeLayer       |       add_dequantize       |
    |        34        |     CONDITION      |       IConditionLayer       |             /              |
    |        35        | CONDITIONAL_INPUT  |  IIfConditionalInputLayer   |             /              |
    |        36        | CONDITIONAL_OUTPUT |  IIfConditionalOutputLayer  |             /              |
    |        37        |      SCATTER       |        IScatterLayer        |        add_scatter         |
    |        38        |       EINSUM       |        IEinsumLayer         |         add_einsum         |
    |        39        |     ASSERTION      |       IAssertionLayer       |       add_assertion        |
    |        40        |      ONE_HOT       |        IOneHotLayer         |        add_one_hot         |
    |        41        |      NON_ZERO      |        INonZeroLayer        |        add_non_zero        |
    |        42        |    GRID_SAMPLE     |      IGridSampleLayer       |      add_grid_sample       |
    |        43        |        NMS         |          INMSLayer          |          add_nms           |
    |        44        |  REVERSE_SEQUENCE  |    IReverseSequenceLayer    |    add_reverse_sequence    |
    |        45        |   NORMALIZATION    |     INormalizationLayer     |     add_normalization      |
    |        46        |     PLUGIN_V3      |       IPluginV3Layer        |       add_plugin_v3        |
    |        47        |      SQUEEZE       |        ISqueezeLayer        |        add_squeeze         |
    |        48        |     UNSQUEEZE      |       IUnsqueezeLayer       |       add_unsqueeze        |
    |        49        |     CUMULATIVE     |      ICumulativeLayer       |       add_cumulative       |
    |        50        |  DYNAMIC_QUANTIZE  |    IDynamicQuantizeLayer    |    add_dynamic_quantize    |
    |        51        |  ATTENTION_INPUT   |    IAttentionInputLayer     |             /              |
    |        52        |  ATTENTION_OUTPUT  |    IAttentionOutputLayer    |             /              |
    |        53        | ROTARY_EMBEDDING   |   IRotaryEmbeddingLayer     |   add_rotary_embedding     |
    |        54        |  KV_CACHE_UPDATE   |    IKVCacheUpdateLayer      |   add_kv_cache_update      |
    |        55        |        MOE         |        IMoELayer            |         add_moe            |
    |        56        |  DIST_COLLECTIVE   |   IDistCollectiveLayer      |    add_dist_collective     |
    |        /         |         /          |              /              |         add_input          |
    |        /         |         /          |     ILoopBoundaryLayer      |          add_loop          |
    |        /         |         /          |   IAttentionBoundaryLayer   |       add_attention        |
    |        /         |         /          | IIfConditionalBoundaryLayer |     add_if_conditional     |
    """
    layer_type_list = sorted(trt.LayerType.__members__)
    layer_name_list = sorted([x for x in dir(trt) if x.endswith("Layer") and x != "ILayer"])
    add_layer_method_name_list = sorted([x for x in dir(trt.INetworkDefinition) if x.startswith("add_") and x != "ILayer"])
    print(layer_type_list)
    print(layer_name_list)
    print(add_layer_method_name_list)

def is_dims_unset(dims: trt.Dims) -> bool:
    """
    Whether a `trt.Dims`-valued layer attribute has never been assigned.

    TensorRT leaves such an attribute at its internal "not set" sentinel `nbDims == -1`, which is
    hostile to read from Python:

    + `len(dims)` raises `ValueError: __len__() should return >= 0`, because the `len()` builtin
      refuses the negative value the binding hands back. So does anything built on it (`list()`,
      `bool()`, iteration, `in`).
    + `repr(dims)` / `str(dims)` do NOT raise. They print a garbage rank, `(80)` or `(81)`
      depending on the TensorRT build - this is the "80/81" that shows up in a VS Code debugger
      watch window while the very same expression throws in a normal script.

    Calling the `__len__` slot directly side-steps the builtin's sign check and returns the raw
    `-1`, so this needs neither a `try`/`except` nor a guess about the garbage rank. Verified
    against `IShuffleLayer.reshape_dims`, `ISliceLayer.axes` / `.start` / `.shape` / `.stride`
    and `I(De)QuantizeLayer.block_shape` on TensorRT 11.1.0.106.
    """
    return dims.__len__() < 0

def layer_type_to_layer_type_name(layer_type: trt.LayerType) -> str:
    """Get layer type name, e.g. LayerType.CONVOLUTION -> "CONVOLUTION"."""
    return layer_type.name
    return str(layer_type)[10:]  # Old method, 10 is hard-code for the length of "LayerType."

def layer_to_layer_class(layer: trt.ILayer = None) -> trt.ILayer:
    """
    Get layer class from input layer
    """
    layer_type_name = layer_type_to_layer_type_name(layer.type)
    # Special cases
    if layer_type_name == "CONDITIONAL_INPUT":
        return trt.IIfConditionalInputLayer
    elif layer_type_name == "CONDITIONAL_OUTPUT":
        return trt.IIfConditionalOutputLayer
    elif layer_type_name == "ELEMENTWISE":
        return trt.IElementWiseLayer
    elif layer_type_name == "LRN":
        return trt.ILRNLayer
    elif layer_type_name == "NMS":
        return trt.INMSLayer
    elif layer_type_name == "KV_CACHE_UPDATE":
        return trt.IKVCacheUpdateLayer
    elif layer_type_name == "MOE":
        return trt.IMoELayer
    elif layer_type_name == "PARAMETRIC_RELU":
        return trt.IParametricReLULayer
    elif layer_type_name == "PLUGIN":
        return None  # IPluginLayer is not supported any more
    elif layer_type_name == "RAGGED_SOFTMAX":
        return trt.IRaggedSoftMaxLayer
    elif layer_type_name == "SOFTMAX":
        return trt.ISoftMaxLayer
    elif layer_type_name == "TOPK":
        return trt.ITopKLayer
    # Normal cases, e.g. MATRIX_MULTIPLY -> MatrixMultiply
    name = "".join(name[0] + name[1:].lower() for name in layer_type_name.split("_"))
    return getattr(trt, f"I{name}Layer")

def layer_dynamic_cast(layer: trt.ILayer = None) -> None:
    """
    Dynamic cast a layer to its real layer type with side effects
    """
    layer.__class__ = layer_to_layer_class(layer)
    return

def layer_type_to_add_layer_method_name(layer_type: trt.LayerType) -> "str":
    """
    Get corresponding `add_*` method for adding the layer
    """
    layer_type_name = layer_type_to_layer_type_name(layer_type)
    # Special cases
    if layer_type_name == "CONDITION":
        return "add_if_conditional"
    elif layer_type_name == "CONVOLUTION":
        return "add_convolution_nd"
    elif layer_type_name == "DECONVOLUTION":
        return "add_deconvolution_nd"
    elif layer_type_name == "GATHER":
        return "add_gather_v2"
    elif layer_type_name == "NORMALIZATION":
        return "add_normalization_v2"
    elif layer_type_name == "PADDING":
        return "add_padding_nd"
    elif layer_type_name == "POOLING":
        return "add_pooling_nd"
    elif layer_type_name == "SCALE":
        return "add_scale_nd"
    # Normal cases, e.g. MATRIX_MULTIPLY -> add_matrix_multiply
    return "add_" + layer_type_name.lower()

########################################################################################################################
# Build a network from an ONNX file

def parse_onnx(
    onnx_file: Union[str, Path] | None = None,
    logger: trt.ILogger | None = None,
    network: trt.INetworkDefinition | None = None,
    builder_config: trt.IBuilderConfig | None = None,
    original_parser: trt.OnnxParser | None = None,
    tw=None,
):
    """Parse an ONNX file into a TensorRT network and print parser errors."""
    if tw is not None:
        logger = tw.logger
        network = tw.network
        builder_config = tw.builder_config
    else:
        assert not (logger is None or network is None or builder_config is None), "Either provide a TRTWrapperV1 or provide builder_config/network/profile separately."
    # Use parser from input argument if exists, otherwise construct a local one
    parser = trt.OnnxParser(network, logger) if original_parser is None else original_parser
    parser.set_builder_config(builder_config)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
    return
