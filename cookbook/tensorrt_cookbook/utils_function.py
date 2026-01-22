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
import re
import sys
from collections import OrderedDict
from pathlib import Path
import ctypes
import numpy as np
import tensorrt as trt
import torch
from numpy.random import default_rng
from cuda.bindings import runtime as cudart
import random

def seed_everything(seed: int = 31193, deterministic: bool = True):
    """Set random seed for reproducible results."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    rng = default_rng(seed)  # Use this in code files
    [rng]  # Avoid `rng` is remove d by yapf
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()

np.set_printoptions(precision=3, linewidth=200, suppress=True)

########################################################################################################################
# Tool functions commonly used

def ceil_divide(a, b):
    return (a + b - 1) // b

def round_up(a, b):
    return ceil_divide(a, b) * b

def byte_to_string(xByte):
    if xByte < (1 << 10):
        return f"{xByte: 5.1f}  B"
    if xByte < (1 << 20):
        return f"{xByte / (1 << 10): 5.1f}KiB"
    if xByte < (1 << 30):
        return f"{xByte / (1 << 20): 5.1f}MiB"
    return f"{xByte / (1 << 30): 5.1f}GiB"

########################################################################################################################
# Tool functions for numpy array

def print_array_information(x: np.array = None, des: str = "", n: int = 5):
    """
    Print statistic information of the tensor `x`
    """
    if 0 in x.shape:
        print('%s:%s' % (des, str(x.shape)))
        return
    x = x.astype(np.float32)
    info = f"{des}:{str(x.shape)},"
    info += f"SumAbs={np.sum(abs(x)):.5e},Var={np.var(x):.5f},"
    info += f"Max={np.max(x):.5f},Min={np.min(x):.5f},"
    info += f"SAD={np.sum(np.abs(np.diff(x.reshape(-1)))):.5f}"
    print(info)
    if n > 0:
        print(" " * len(des) + "   ", x.reshape(-1)[:n], x.reshape(-1)[-n:])
    return

def check_array(a, b, weak=False, des="", error_epsilon=1e-5):
    """
    Compare tensor `a` and `b`
    """
    if a.shape != b.shape:
        print(f"[check]Shape different: A{a.shape} : B{b.shape}")
        return
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < error_epsilon)
    else:
        if a.dtype == bool:
            a = a.astype(np.int32)
        if b.dtype == bool:
            b = b.astype(np.int32)
        res = np.all(a == b)
    maxAbsDiff = np.max(np.abs(a - b))
    meanAbsDiff = np.mean(np.abs(a - b))
    maxRelDiff = np.max(np.abs(a - b) / (np.abs(b) + error_epsilon))
    meanRelDiff = np.mean(np.abs(a - b) / (np.abs(b) + error_epsilon))
    result = f"[check]{des}:{res},{maxAbsDiff=:.2e},{meanAbsDiff=:.2e},{maxRelDiff=:.2e},{meanRelDiff=:.2e}"

    index = np.argmax(np.abs(a - b))
    valueA, valueB = a.flatten()[index], b.flatten()[index]
    shape = a.shape
    indexD = []
    for i in range(len(shape) - 1, -1, -1):
        x = index % shape[i]
        indexD = [x] + indexD
        index = index // shape[i]
    result += f"\n    worstPair=({valueA}:{valueB})@{indexD}"
    print(result)
    return res

########################################################################################################################
# Data type conversion functions, copy from TensorRT-LLM/tensorrt_llm/_utils.py

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})

_datatype_str_to_np = dict(
    bfloat16=np_bfloat16,
    bool=np.bool_,
    float16=np.float16,
    float32=np.float32,
    fp8=np_float8,
    int32=np.int32,
    int64=np.int64,
    int8=np.int8,
    fp32=np.float32,  # Other alias
    float=np.float32,
    fp16=np.float16,
    half=np.float16,
)

def datatype_str_to_np(dtype: str):
    ret = _datatype_str_to_np.get(dtype.lower())
    assert ret is not None, f'Unsupported data type: {dtype}'
    return ret

_datatype_np_to_str = {  # Do not use reverse map to avoid duplicate keys
    np_bfloat16: "bfloat16",
    np_float8: "fp8",
    np.bool_: "bool",
    np.float16: "float16",
    np.float32: "float32",
    np.int32: "int32",
    np.int64: "int64",
    np.int8: "int8",
}

def datatype_np_to_str(dtype: np.dtype):
    ret = _datatype_np_to_str.get(dtype)
    assert ret is not None, f'Unsupported data type: {dtype}'
    return ret

_datatype_str_to_torch = dict(
    bfloat16=torch.bfloat16,
    bool=torch.bool,
    float16=torch.float16,
    float32=torch.float32,
    fp8=torch.float8_e4m3fn,
    int32=torch.int32,
    int64=torch.int64,
    int8=torch.int8,
)

def datatype_str_to_torch(dtype: str):
    ret = _datatype_str_to_torch.get(dtype.lower())
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_torch_to_str = {v: k for k, v in _datatype_str_to_torch.items()}

def datatype_torch_to_str(dtype: torch.dtype):
    return _datatype_torch_to_str[dtype]

_datatype_str_to_trt = dict(
    bfloat16=trt.bfloat16,
    bool=trt.bool,
    float16=trt.float16,
    float32=trt.float32,
    fp8=trt.fp8,
    int32=trt.int32,
    int64=trt.int64,
    int8=trt.int8,
    nvfp4=trt.fp4,
)

def datatype_str_to_trt(dtype: str):
    ret = _datatype_str_to_trt.get(dtype.lower())
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_trt_to_str = {v: k for k, v in _datatype_str_to_trt.items()}

def datatype_trt_to_str(dtype: trt.DataType) -> str:
    assert isinstance(dtype, trt.DataType)
    return _datatype_trt_to_str[dtype]

_datatype_np_to_trt = {
    np_bfloat16: trt.bfloat16,
    np_float8: trt.fp8,
    np.bool_: trt.bool,
    np.float16: trt.float16,
    np.float32: trt.float32,
    np.int32: trt.int32,
    np.int64: trt.int64,
    np.int8: trt.int8,
    np.uint8: trt.uint8,
    np.dtype('bool'): trt.bool,  # hash of np.dtype('bool') != np.bool_
    np.dtype('float16'): trt.float16,
    np.dtype('float32'): trt.float32,
    np.dtype('int32'): trt.int32,
    np.dtype('int64'): trt.int64,
    np.dtype('int8'): trt.int8,
}

def datatype_np_to_trt(dtype: np.dtype):
    ret = _datatype_np_to_trt.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_trt_to_np = {  # Do not use reverse map to avoid duplicate keys
    trt.bfloat16: np_bfloat16,
    trt.bool: np.bool_,
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.fp8: np_float8,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.int8: np.int8,
    trt.uint8: np.uint8,
}
# Data type in TensorRT but not in numpy: trt.e8m0, trt.fp4, trt.int4

def datatype_trt_to_np(dtype: trt.DataType):
    ret = _datatype_trt_to_np.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_torch_to_np = {
    torch.bfloat16: np_bfloat16,
    torch.bool: np.bool_,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float8_e4m3fn: np_float8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.complex128: np.complex128,  # data types we do not use here
    torch.complex64: np.complex64,
    torch.uint16: np.uint16,
    torch.uint32: np.uint32,
    torch.uint64: np.uint64,
}
# Data type in torch but not in numpy:
#    torch.complex32
#    torch.float8_e4m3fn
#    torch.float8_e4m3fnuz
#    torch.float8_e4m3fnuz
#    torch.float8_e5m2
#    torch.float8_e5m2fnuz
#    torch.float8_e5m2fnuz
#    torch.float8_e8m0fnu
#    torch.qint2x4
#    torch.qint32
#    torch.qint8
#    torch.quint4x2
#    torch.quint8

def datatype_torch_to_np(dtype: torch.dtype):
    ret = _datatype_torch_to_np.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_np_to_torch = {v: k for k, v in _datatype_torch_to_np.items()}
# Data type in numpy but not in torch:
#    numpy.float128
#    nnumpy.complex256

def datatype_np_to_torch(dtype: np.dtype):
    ret = _datatype_np_to_torch.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_trt_to_torch = {
    trt.bfloat16: torch.bfloat16,
    trt.bool: torch.bool,
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.fp8: torch.float8_e4m3fn,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.e8m0: torch.float8_e8m0fnu,
    trt.int4: torch.int4,
    trt.uint8: torch.uint8,
}
# Data type in TensorRT but not in torch:  trt.fp4

def datatype_trt_to_torch(dtype: trt.DataType):
    ret = _datatype_trt_to_torch.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_torch_to_trt = {v: k for k, v in _datatype_trt_to_torch.items()}

def datatype_torch_to_trt(dtype: torch.dtype):
    ret = _datatype_torch_to_trt.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_torch_to_np_typestr = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int8: "|i1",
    torch.float8_e4m3fn: "|i1",
    torch.qint8: "|u1",
    torch.bool: "|b1",
    torch.bfloat16: "<f2",
    torch.uint8: "|u1",
}

def datatype_torch_to_np_typestr(dtype: torch.dtype):
    ret = _datatype_torch_to_np_typestr.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_np_to_trt_field = {
    np_bfloat16: trt.PluginFieldType.BF16,
    np.float16: trt.PluginFieldType.FLOAT16,
    np.float32: trt.PluginFieldType.FLOAT32,
    np.float64: trt.PluginFieldType.FLOAT64,
    np.int16: trt.PluginFieldType.INT16,
    np.int32: trt.PluginFieldType.INT32,
    np.int64: trt.PluginFieldType.INT64,
    np.int8: trt.PluginFieldType.INT8,
    np.dtype('float16'): trt.PluginFieldType.FLOAT16,  # hash of np.dtype('float16') != np.float16
    np.dtype('float32'): trt.PluginFieldType.FLOAT32,
    np.dtype('float64'): trt.PluginFieldType.FLOAT64,
    np.dtype('int16'): trt.PluginFieldType.INT16,
    np.dtype('int32'): trt.PluginFieldType.INT32,
    np.dtype('int64'): trt.PluginFieldType.INT64,
    np.dtype('int8'): trt.PluginFieldType.INT8,
}
# Data type in trt.PluginFieldType but not in numpy:
# trt.PluginFieldType.CHAR
# trt.PluginFieldType.DIMS
# trt.PluginFieldType.FP4
# trt.PluginFieldType.FP8
# trt.PluginFieldType.INT4

def datatype_np_to_trt_pluginfield(dtype: np.dtype) -> trt.PluginFieldType:
    ret = _datatype_np_to_trt_field.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

_datatype_trt_field_to_np = {
    trt.PluginFieldType.BF16: np_bfloat16,
    trt.PluginFieldType.CHAR: np.int8,
    trt.PluginFieldType.FLOAT16: np.float16,
    trt.PluginFieldType.FLOAT32: np.float32,
    trt.PluginFieldType.FLOAT64: np.float64,
    trt.PluginFieldType.INT16: np.int16,
    trt.PluginFieldType.INT32: np.int32,
    trt.PluginFieldType.INT64: np.int64,
    trt.PluginFieldType.INT8: np.int8,
}

def datatype_trt_pluginfield_to_np(dtype: trt.PluginFieldType) -> np.dtype:
    ret = _datatype_trt_field_to_np.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

def format_to_string(format_bit_mask):
    """
    Get format description from format bit
    """
    output = ""
    if format_bit_mask & (1 << int(trt.TensorFormat.LINEAR)):  # 0
        output += "LINEAR,"
    if format_bit_mask & (1 << int(trt.TensorFormat.CHW2)):  # 1
        output += "CHW2,"
    if format_bit_mask & (1 << int(trt.TensorFormat.HWC8)):  # 2
        output += "HWC8,"
    if format_bit_mask & (1 << int(trt.TensorFormat.CHW4)):  # 3
        output += "CHW4,"
    if format_bit_mask & (1 << int(trt.TensorFormat.CHW16)):  # 4
        output += "CHW16,"
    if format_bit_mask & (1 << int(trt.TensorFormat.CHW32)):  # 5
        output += "CHW32,"
    if format_bit_mask & (1 << int(trt.TensorFormat.DHWC8)):  # 6
        output += "DHWC8,"
    if format_bit_mask & (1 << int(trt.TensorFormat.CDHW32)):  # 7
        output += "CDHW32,"
    if format_bit_mask & (1 << int(trt.TensorFormat.HWC)):  # 8
        output += "HWC,"
    if format_bit_mask & (1 << int(trt.TensorFormat.DLA_LINEAR)):  # 9
        output += "DLA_LINEAR,"
    if format_bit_mask & (1 << int(trt.TensorFormat.DLA_HWC4)):  # 10
        output += "DLA_HWC4,"
    if format_bit_mask & (1 << int(trt.TensorFormat.HWC16)):  # 11
        output += "DHWC16,"
    if format_bit_mask & (1 << int(trt.TensorFormat.DHWC)):  # 12
        output += "DHWC,"
    if len(output) == 0:
        output = "None"
    else:
        output = output[:-1]
    return output

def torch_to_numpy(x: torch.Tensor):
    assert isinstance(x, torch.Tensor), f'x must be a torch.Tensor object, but got {type(x)}.'
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).detach().cpu().numpy().view(np_bfloat16)
    elif x.dtype == torch.float8_e4m3fn:
        return x.view(torch.int8).detach().cpu().numpy().view(np_float8)
    return x.detach().cpu().numpy()

def torch_to_numpy(x: torch.Tensor, ndarray: np.array):
    if x.dtype == torch.bfloat16:
        torch.from_numpy(ndarray.view(np.int16)).copy_(x.view(torch.int16))
    elif x.dtype == torch.float8_e4m3fn:
        torch.from_numpy(ndarray.view(np.int8)).copy_(x.view(torch.int8))
    else:
        torch.from_numpy(ndarray).copy_(x)
    return ndarray

def numpy_to_torch(x):
    if x.dtype == np_bfloat16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    elif x.dtype == np_float8:
        return torch.from_numpy(x.view(np.int8)).view(torch.float8_e4m3fn)
    return torch.from_numpy(x)

def numpy_as_dtype(x, dtype: str):
    if datatype_str_to_np(dtype) == x.dtype:
        return x
    if x.dtype not in [np_bfloat16, np_float8] and dtype not in ['bfloat16', 'fp8']:
        return x.astype(datatype_str_to_np(dtype))
    else:
        return torch_to_numpy(numpy_to_torch(x).to(datatype_str_to_torch(dtype)))

########################################################################################################################
# Tool functions related to TensorRT

def text_to_logger_level(level):
    if level.upper() == "VERBOSE":  # Use `match-case` when yapf supports
        return trt.Logger.Severity.VERBOSE
    elif level.upper() == "INFO":
        return trt.Logger.Severity.INFO
    elif level.upper() == "WARNING":
        return trt.Logger.Severity.WARNING
    elif level.upper() == "ERROR":
        return trt.Logger.Severity.ERROR
    elif level.upper() in ["INTERNAL_ERROR", "INTERNAL"]:
        return trt.Logger.Severity.INTERNAL_ERROR
    else:
        print(f"Error log level {level}, set to ERROR")
        return trt.Logger.Severity.ERROR

def print_layer_class():
    """
    Layer name map in TensorRT-10.14.1.48:
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
    |        50        | DYNAMIC_QUANTIZE'  |    IDynamicQuantizeLayer    |    add_dynamic_quantize    |
    |        51        |  ATTENTION_INPUT   |    IAttentionInputLayer     |             /              |
    |        52        |  ATTENTION_OUTPUT  |    IAttentionOutputLayer    |             /              |
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

def layer_type_to_layer_type_name(layer_type: trt.LayerType) -> str:
    """
    Get layer type name
    e.g. <LayerType.CONVOLUTION: 0> -> 'LayerType.CONVOLUTION' -> 'CONVOLUTION'
    """
    return str(layer_type)[10:]  # 10 is hard-code for the length of "LayerType."

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
    elif layer_type_name == "PADDING":
        return "add_padding_nd"
    elif layer_type_name == "POOLING":
        return "add_pooling_nd"
    elif layer_type_name == "SCALE":
        return "add_scale_nd"
    # Normal cases, e.g. MATRIX_MULTIPLY -> add_matrix_multiply
    return "add_" + layer_type_name.lower()

########################################################################################################################
# Tool functions related to Cookbook utilities

def case_mark(f):
    """
    Wrapper of cookbook example case
    """

    def f_with_mark(*args, **kargs):
        print("=" * 30 + f" Start [{f.__name__},{args},{kargs}]")
        result = f(*args, **kargs)
        print("=" * 30 + f" End   [{f.__name__}]")
        return result

    return f_with_mark

class Pointer:

    def __init__(self, byte):
        self.byte = byte
        self.offset = 0

    def set_offset(self, offset):
        self.offset = offset

    def read_and_move(self, size: int = 1, return_number: bool = True):
        target_byte = self.byte[self.offset:self.offset + size]
        self.offset += size
        if return_number:
            return int.from_bytes(target_byte, byteorder=sys.byteorder)
        return target_byte

    def f(self, name, size):  # Print and return value
        data_number = self.read_and_move(size)
        if name != "":
            print(f"{name:<28s}:{data_number:>16d}")
        return data_number

def print_engine_information(
    trt_file: Path = Path(),
    plugin_file_list: list = [],
    device_index: list[int] = [0],
):

    logger = trt.Logger()
    trt.init_libnvinfer_plugins(logger, namespace="")

    for plugin_file in plugin_file_list:
        if plugin_file.exists():
            ctypes.cdll.LoadLibrary(plugin_file)

    with open(trt_file, "rb") as f:
        engine_bytes = f.read()
    p = Pointer(engine_bytes)

    logger.log(trt.Logger.INFO, "This function is verified in TRT-10.14.1.48, it might not work on other TRT version.")
    print("=" * 64 + " Current TensorRT")  # Print current TRT environment
    info = ""
    for path in os.popen("find /usr -name NvInferVersion.h"):
        try:
            with open(path[:-1], "r") as f:
                info = f.read()
                break
        except:
            continue
    if info == "":
        print("Fail finding TensorRT library")
        exit()

    v_major = re.search(r"NV_TENSORRT_MAJOR \d+", info) or re.search(r"TRT_MAJOR_ENTERPRISE \d+", info)
    v_minor = re.search(r"NV_TENSORRT_MINOR \d+", info) or re.search(r"TRT_MINOR_ENTERPRISE \d+", info)
    v_patch = re.search(r"NV_TENSORRT_PATCH \d+", info) or re.search(r"TRT_PATCH_ENTERPRISE \d+", info)
    v_build = re.search(r"NV_TENSORRT_BUILD \d+", info) or re.search(r"TRT_BUILD_ENTERPRISE \d+", info)
    v_major = "None" if v_major is None else v_major.group().split(" ")[-1]
    v_minor = "None" if v_minor is None else v_minor.group().split(" ")[-1]
    v_patch = "None" if v_patch is None else v_patch.group().split(" ")[-1]
    v_build = "None" if v_build is None else v_build.group().split(" ")[-1]
    print(f"{'Major':<28s}:{v_major:>16s}")
    print(f"{'Minor':<28s}:{v_minor:>16s}")
    print(f"{'Patch':<28s}:{v_patch:>16s}")
    print(f"{'Build':<28s}:{v_build:>16s}")
    print(f"{'TRT-Python':<28s}:{trt.__version__:>16s}")

    # ================================================================
    print("=" * 64 + " Engine header")
    print(f"{'HeaderSize':<28s}:{32:>16d}")
    p.f("MagicTag", 4)  # uint32_t -> 4
    p.f("SerializationVersion", 4)
    nEntry = p.f("nEntry", 8)  # uint64_t -> 8
    p.f("PlanTotalSize", 8)
    trt_major = p.f("TRT.Major", 1)  # uint8_t -> 1
    trt_minor = p.f("TRT.Minor", 1)
    p.f("TRT.Patch", 1)
    p.f("TRT.Build", 1)
    p.f("Pad", 4)

    # ================================================================
    print("=" * 64 + " Engine data")

    p.set_offset(32)  # Skip head, in fact `p` must be already at this location before this expression
    section_list = []
    for i in range(nEntry):
        type = p.read_and_move(4, False)
        pad = p.read_and_move(4)
        offset = p.read_and_move(8)
        size = p.read_and_move(8)
        section_list.append([type, pad, offset, size])
    #print(section_list)

    p.set_offset(section_list[0][2])  # We only print engine section (index `0`)
    p.f("MagicTag", 4)
    if not (trt_major >= 10 and trt_minor >= 6):
        p.f("SafeVersion", 4)
    p.f("StdVersion", 4)
    p.f("HashRead", 4)
    p.f("SizeRead", 8)
    p.f("", 4)
    p.f("TRT.Major", 1)
    p.f("", 4)
    p.f("TRT.Minor", 1)
    p.f("", 4)
    p.f("TRT.Patch", 1)
    p.f("", 4)
    p.f("TRT.Build", 1)
    p.f("", 8)
    p.f("HardwareCompatLevel", 4)
    p.f("", 8)
    p.f("", 4)

    # ================================================================
    print("=" * 64 + " Device information")
    print(f"{'Property name':<28s}:{'Engine':^16s} <-> {'Current':^16s}")

    # `eci` for Engine CUDA Information, but name is aligned with `cudart.cudaGetDeviceProperties`
    p.offset = 144
    eci = OrderedDict()
    eci["totalGlobalMem"] = p.f("", 8)
    p.f("", 4)
    eci["l2CacheSize"] = p.f("", 8)
    p.f("", 4)
    eci["persistingL2CacheMaxSize"] = p.f("", 8)
    p.f("", 4)
    eci["clockRate"] = p.f("", 4)
    p.f("", 4)
    eci["memoryClockRate"] = p.f("", 4)
    p.f("", 4)
    eci["memoryBusWidth"] = p.f("", 4)
    p.f("", 4)
    eci["sharedMemPerBlock"] = p.f("", 4)
    p.f("", 4)
    eci["sharedMemPerMultiprocessor"] = p.f("", 4)
    p.f("", 4)
    eci["multiProcessorCount"] = p.f("", 4)
    p.f("", 4)
    eci["integrated"] = p.f("", 4)
    p.f("", 4)
    eci["maxThreadsPerBlock"] = p.f("", 4)
    p.f("", 4)
    eci["reservedSharedMemPerBlock"] = p.f("", 4)
    p.f("", 4)
    eci["major"] = p.f("", 4)
    p.f("", 4)
    eci["minor"] = p.f("", 4)
    p.f("", 4)
    eci["textureAlignment"] = p.f("", 4)
    p.f("", 4)
    _, info = cudart.cudaGetDeviceProperties(device_index)
    # `rci` for Runtime CUDA Information
    rci = OrderedDict()
    for name in dir(info):
        rci[name] = getattr(info, name, None)

    for name in eci:
        print(f"{name:<28s}:{eci[name]:16d} <->{rci.get(name, -1):16d}")

    return

def print_engine_io_information(
    *,
    trt_file: Path = Path(),
    engine: trt.ICudaEngine = None,
    plugin_file_list: list = [],
) -> None:
    if engine is None:
        with open(trt_file, "rb") as f:
            engine_bytes = f.read()

        logger = trt.Logger(trt.Logger.Severity.ERROR)
        # Load TenorRT native and customer's plugins
        trt.init_libnvinfer_plugins(logger, namespace="")
        for plugin_file in plugin_file_list:
            if plugin_file.exists():
                ctypes.cdll.LoadLibrary(plugin_file)
        runtime = trt.Runtime(logger)
        try:
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        except:
            print("Failed loading engine, `print_engine_io_information()` is only supported when TRT version of engine and runtime is the same.")
            return

    context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    n_optimization_profile = engine.num_optimization_profiles
    max_name_width = 8  # Maximum Width of tensor Name
    max_shape_width = 0  # Maximum Width of tensor Shape

    # Get information of engine input / output
    tid = {}  # Tensor Information Dictionary
    for name in tensor_name_list:
        tensor = dict()
        max_name_width = max(max_name_width, len(name))
        tensor["mode"] = 'I' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'O'
        tensor["location"] = 'GPU' if engine.get_tensor_location(name) else 'CPU'
        tensor["data_type"] = str(engine.get_tensor_dtype(name))[9:]
        tensor["build_shape"] = str(engine.get_tensor_shape(name))
        tensor["profile_list"] = [[] for _ in range(n_optimization_profile)]
        if tensor["mode"] == "I":
            for i in range(n_optimization_profile):
                if tensor["location"] == "GPU":
                    shape = engine.get_tensor_profile_shape(name, i)
                else:
                    shape = engine.get_tensor_profile_value(i, name)
                tensor["profile_list"][i].extend(shape)
                max_shape_width = max(max_shape_width, *[len(str(s)) for s in shape])
        tid[name] = tensor

    # Set input shape to get output shape
    for i in range(n_optimization_profile):
        for j in range(3):  # Min, Opt, Max
            for name in tid.keys():
                if tid[name]["mode"] == "I":
                    if tid[name]["location"] == "GPU":
                        context.set_input_shape(name, tid[name]["profile_list"][i][j])
                    else:
                        context.set_tensor_address(name, tid[name]["profile_list"][i][j].ctypes.data)
                elif tid[name]["mode"] == "O":
                    assert context.all_binding_shapes_specified and context.all_shape_inputs_specified
                    shape = context.get_tensor_shape(name)
                    tid[name]["profile_list"][i].append(shape)
                    max_shape_width = max(max_shape_width, len(str(shape)))

    print("\nInformation of engine input / output.")
    print(f"{'='*(max_name_width + max_shape_width + 24)}")
    print(f"{'Name':^{max_name_width}}|I/O|Location|DataType|{'Shape':^{max_shape_width}}|")
    print(f"{'-'*(max_name_width + max_shape_width + 24)}")
    for name in tensor_name_list:
        tensor = tid[name]
        info = f"{name:<{max_name_width}}|{tensor['mode']:^3s}|{tensor['location']:^8s}|{tensor['data_type']:^8s}|"
        info += f"{tensor['build_shape']:^{max_shape_width}}|"
        print(info)
    print(f"{'='*(max_name_width + max_shape_width + 24)}")

    print("\nInformation of optimization profile.")
    for i in range(n_optimization_profile):
        print(f"\nOptimization Profile {i}:")
        print(f"{'='*(max_name_width + max_shape_width * 3 + 4)}")
        print(f"{'Name':^{max_name_width}}|{'Min':^{max_shape_width}}|{'Opt':^{max_shape_width}}|{'Max':^{max_shape_width}}|")
        print(f"{'-'*(max_name_width + max_shape_width * 3 + 4)}")
        for name in tensor_name_list:
            tensor = tid[name]
            info = f"{name:<{max_name_width}}|"
            info += f"{str(tensor['profile_list'][i][0]):^{max_shape_width}}|"
            info += f"{str(tensor['profile_list'][i][1]):^{max_shape_width}}|"
            info += f"{str(tensor['profile_list'][i][2]):^{max_shape_width}}|"
            print(info)
        print(f"{'='*(max_name_width + max_shape_width * 3 + 4)}")
    return

def print_context_io_information(
    engine: trt.ICudaEngine = None,
    context: trt.IExecutionContext = None,
    context_index: int = 0,
) -> None:
    n_io = engine.num_io_tensors
    max_name_width = 8  # Maximum Width of tensor Name
    max_shape_width = 0  # Maximum Width of tensor Shape
    tensorInfo = {}
    for i in range(n_io):
        name = engine.get_tensor_name(i)
        b_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = str(engine.get_tensor_shape(name))
        tensorInfo[i] = [name, b_input, shape]
        max_name_width = max(max_name_width, len(name))
        max_shape_width = max(max_shape_width, len(shape))
        # Shape input tensor is not used in TRT-LLM yet

    print(f"Information of context input / output.")
    print(f"Using Optimization Profile: {context_index}")
    print(f"{'='*(max_name_width + max_shape_width + 6)}")
    print(f"{'Name':^{max_name_width}}|I/O|{'Shape':^{max_shape_width}}|")
    print(f"{'-'*(max_name_width + max_shape_width + 6)}")
    for i in range(n_io):
        name, b_input, shape = tensorInfo[i]
        info = f"{name:<{max_name_width}}|{'I' if b_input else 'O':^3s}|{shape:^{max_shape_width}}|"
        print(info)
    print(f"{'='*(max_name_width + max_shape_width + 6)}")

    return
