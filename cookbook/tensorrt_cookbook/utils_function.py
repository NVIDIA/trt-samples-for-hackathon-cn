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

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart

np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()
torch.manual_seed(31193)
torch.cuda.manual_seed_all(31193)
torch.backends.cudnn.deterministic = True

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

def datatype_trt_to_string(datatype_trt: trt.DataType) -> str:
    """
    Cast TensorRT data type into string
    """
    assert isinstance(datatype_trt, trt.DataType), f"Data type `{datatype_trt}` is not a supported data type in TensorRT"
    return str(datatype_trt)[9:]

def datatype_trt_to_torch(datatype_trt: trt.DataType):
    """
    Cast TensorRT data type into Torch
    """
    assert isinstance(datatype_trt, trt.DataType), f"Data type `{datatype_trt}` is not a supported data type in TensorRT"
    if datatype_trt == trt.float32:
        return torch.float32
    if datatype_trt == trt.float16:
        return torch.float16
    if datatype_trt == trt.int8:
        return torch.int8
    if datatype_trt == trt.int32:
        return torch.int32
    if datatype_trt == trt.bool:
        return torch.bool
    if datatype_trt == trt.uint8:
        return torch.uint8
    if datatype_trt == trt.DataType.FP8:
        return torch.float8_e4m3fn
    if datatype_trt == trt.bf16:
        return torch.bfloat16
    if datatype_trt == trt.int64:
        return torch.int64
    if datatype_trt == trt.int4:  # only torch.uint4 is supported
        print(f"Data type `{datatype_trt_to_string(datatype_trt)}` is not supported in pyTorch")
        return None
    assert False, f"Data type `{datatype_trt_to_string(datatype_trt)}` is not supported in Cookbook yet"
    return None

def datatype_np_to_trt(datatype_np: np.dtype) -> trt.DataType:
    """
    Cast TensorRT data type into Torch
    """
    assert isinstance(datatype_np, np.dtype), f"Data type  `{datatype_np}` is not a supported data type in numpy"
    if datatype_np == np.float32:
        return trt.float32
    if datatype_np == np.float16:
        return trt.float16
    if datatype_np == np.int8:
        return trt.int8
    if datatype_np == np.int32:
        return trt.int32
    if datatype_np == bool:
        return trt.bool
    if datatype_np == np.uint8:
        return trt.uint8
    if datatype_np == np.int64:
        return trt.int64
    assert False, f"Data type `{datatype_np}` is not supported in Cookbook yet"
    return None

def datatype_engine_to_string(string: str = ""):
    """
    Cast TensorRT engine data type into string
    """
    if string in ["FP32", "Float"]:
        return np.float32
    elif string in ["FP16", "Half"]:
        return np.float16
    elif string in ["INT8", "Int8"]:
        return np.int8
    elif string in ["Int32"]:
        return np.int32
    elif string in ["BOOL", "Bool"]:
        return bool
    elif string in ["UInt8"]:  # "UINT8"
        return np.uint8
    elif string in ["FP8"]:
        return "FP8"
    elif string in ["BFloat16"]:
        return "BF16"
    elif string in ["Int64"]:
        return np.int64
    elif string in ["Int4"]:
        return "INT4"
    assert False, f"Data type `{datatype_np}` is not supported in Cookbook yet"
    return None

def layer_type_to_class(layer: trt.ILayer = None) -> trt.ILayer:
    """
    Get layer class from input layer
    """
    layer_type_name = str(layer.type)[10:]
    # Some special cases
    if layer_type_name == "ELEMENTWISE":
        return trt.IElementWiseLayer
    if layer_type_name == "LRN":
        return trt.ILRNLayer
    if layer_type_name == "NMS":
        return trt.INMSLayer
    if layer_type_name == "PARAMETRIC_RELU":
        return trt.IParametricReLULayer
    if layer_type_name == "PLUGIN":
        return None  # IPluginLayer is not supported any more
    if layer_type_name == "RAGGED_SOFTMAX":
        return trt.IRaggedSoftMaxLayer
    if layer_type_name == "SOFTMAX":
        return trt.ISoftMaxLayer
    if layer_type_name == "TOPK":
        return trt.ITopKLayer

    name = "".join(name[0] + name[1:].lower() for name in layer_type_name.split("_"))  # e.g. MATRIX_MULTIPLY -> MatrixMultiply
    return trt.__builtins__["getattr"](trt, f"I{name}Layer")

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
        res = np.all(a == b)
    maxAbsDiff = np.max(np.abs(a - b))
    meanAbsDiff = np.mean(np.abs(a - b))
    maxRelDiff = np.max(np.abs(a - b) / (np.abs(b) + error_epsilon))
    meanRelDiff = np.mean(np.abs(a - b) / (np.abs(b) + error_epsilon))
    result = f"[check]{des}:{res},{maxAbsDiff=:.2e},{meanAbsDiff=:.2e},{maxRelDiff=:.2e},{meanRelDiff=:.2e}"
    if maxAbsDiff > error_epsilon:
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

def case_mark(f):
    """
    Wrapper of cookbook example case
    """

    def f_with_mark(*args, **kargs):
        print("=" * 30 + f" Start [{f.__name__},{args},{kargs}]")
        f(*args, **kargs)
        print("=" * 30 + f" End   [{f.__name__}]")

    return f_with_mark
