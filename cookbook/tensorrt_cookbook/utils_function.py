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

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

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

def print_engine_information(trt_file: Path = Path(), plugin_file_list: list = [], device_index: list[int] = [0]):

    with open(trt_file, "rb") as f:
        engine_bytes = f.read()
    p = Pointer(engine_bytes)

    # ================================================================
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

    v_major = re.search(r"NV_TENSORRT_MAJOR \d+", info)
    v_minor = re.search(r"NV_TENSORRT_MINOR \d+", info)
    v_patch = re.search(r"NV_TENSORRT_PATCH \d+", info)
    v_build = re.search(r"NV_TENSORRT_BUILD \d+", info)
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

    # `eci` for Engine CUDA Information
    eci = OrderedDict()
    eci["major"] = p.f("", 4)
    p.f("", 4)
    eci["minor"] = p.f("", 4)
    p.f("", 4)
    eci["maxCoreClockRate"] = p.f("", 4)
    p.f("", 4)
    eci["maxMemoryClockRate"] = p.f("", 4)
    p.f("", 4)
    eci["memoryBusWidth"] = p.f("", 4)
    p.f("", 4)
    eci["l2CacheSize"] = p.f("", 4)
    p.f("", 8)
    eci["maxPersistentL2CacheSize"] = p.f("", 4)
    p.f("", 8)
    eci["sharedMemPerBlock"] = p.f("", 4)
    p.f("", 4)
    eci["sharedMemPerMultiprocessor"] = p.f("", 4)
    p.f("", 4)
    eci["textureAlignment"] = p.f("", 4)
    p.f("", 4)
    eci["multiProcessorCount"] = p.f("", 4)
    p.f("", 4)
    eci["integrated"] = p.f("", 1)
    p.f("", 4)
    eci["maxThreadsPerBlock"] = p.f("", 4)
    p.f("", 4)
    eci["maxGridDimX"] = p.f("", 4)
    p.f("", 4)
    eci["maxGridDimY"] = p.f("", 4)
    p.f("", 4)
    eci["maxGridDimZ"] = p.f("", 4)
    p.f("", 4)
    if trt_major >= 10:
        p.f("", 8)
    eci["totalGlobalMem"] = p.f("", 8)
    p.f("", 4)
    eci["maxTexture1DLinear"] = p.f("", 4)
    p.f("", 4)

    _, info = cudart.cudaGetDeviceProperties(device_index)
    # `rci` for Runtime CUDA Information
    rci = OrderedDict()
    rci["major"] = info.major
    rci["minor"] = info.minor
    rci["maxCoreClockRate"] = info.clockRate
    rci["maxMemoryClockRate"] = info.memoryClockRate
    rci["memoryBusWidth"] = info.memoryBusWidth
    rci["l2CacheSize"] = info.l2CacheSize
    rci["maxPersistentL2CacheSize"] = info.persistingL2CacheMaxSize
    rci["sharedMemPerBlock"] = info.sharedMemPerBlock
    rci["sharedMemPerMultiprocessor"] = info.sharedMemPerMultiprocessor
    rci["textureAlignment"] = info.textureAlignment
    rci["multiProcessorCount"] = info.multiProcessorCount
    rci["integrated"] = info.integrated
    rci["maxThreadsPerBlock"] = info.maxThreadsPerBlock
    rci["maxGridDimX"] = info.maxGridSize[0]
    rci["maxGridDimY"] = info.maxGridSize[1]
    rci["maxGridDimZ"] = info.maxGridSize[2]
    rci["totalGlobalMem"] = info.totalGlobalMem
    rci["maxTexture1DLinear"] = info.maxTexture1DLinear

    for name in eci.keys():
        print(f"{name:<28s}:{eci[name]:16d} <->{rci[name]:16d}")

    return

def print_io_information(trt_file: Path = Path(), plugin_file_list: list = []):

    with open(trt_file, "rb") as f:
        engine_bytes = f.read()

    print("=" * 64 + " Input / output information")

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
        print("Failed loading engine, `print_io_information()` is only supported when TRT version of engine and runtime is the same.")
        return
    context = engine.create_execution_context()

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    max_name_width = 4  # width of 'Name' this word
    max_shape_width = 0
    n_optimization_profile = engine.num_optimization_profiles
    tld = {}  # Tensor Information Dictionary
    # Get basic information
    for name in tensor_name_list:
        dd = dict()
        max_name_width = max(max_name_width, len(name))
        dd["mode"] = 'I' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'O'
        dd["location"] = 'GPU' if engine.get_tensor_location(name) else 'CPU'
        dd["data_type"] = str(engine.get_tensor_dtype(name))[9:]
        dd["build_shape"] = str(engine.get_tensor_shape(name))
        dd["profile_list"] = [[] for _ in range(n_optimization_profile)]
        if dd["mode"] == "I":
            for i in range(n_optimization_profile):
                if engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                    shape = engine.get_tensor_profile_shape(name, i)
                else:
                    shape = engine.get_tensor_profile_value(i, name)
                dd["profile_list"][i].extend(shape)
                max_shape_width = max(max_shape_width, *[len(str(s)) for s in shape])
        tld[name] = dd

    # Set input shape to get output shape
    for i in range(n_optimization_profile):
        for j in range(3):  # Min, Opt, Max
            for name in tld.keys():
                if tld[name]["mode"] == "I":
                    if tld[name]["location"] == "GPU":
                        context.set_input_shape(name, tld[name]["profile_list"][i][j])
                    else:
                        context.set_tensor_address(name, tld[name]["profile_list"][i][j].ctypes.data)
                elif tld[name]["mode"] == "O":
                    assert context.all_binding_shapes_specified and context.all_shape_inputs_specified
                    shape = context.get_tensor_shape(name)
                    tld[name]["profile_list"][i].append(shape)

    # Print basic input / output information
    print(f"{'Name':^{max_name_width}}|I/O|Location|DataType|{'Shape':^{max_shape_width}}|")
    for name in tensor_name_list:
        item = tld[name]
        info = f"{name:<{max_name_width}}|{item['mode']:^3s}|{item['location']:^8s}|{item['data_type']:^8s}|"
        info += f"{item['build_shape']:^{max_shape_width}}|"
        print(info)

    # Print optimization profile information
    for i in range(engine.num_optimization_profiles):
        print(f"\nOptimization Profile {i}:")
        print(f"{'-'*(max_name_width + max_shape_width * 3 + 4)}")
        print(f"{'Name':^{max_name_width}}|{'Min':^{max_shape_width}}|{'Opt':^{max_shape_width}}|{'Max':^{max_shape_width}}|")
        for name in tensor_name_list:
            item = tld[name]
            info = f"{name:<{max_name_width}}|"
            info += f"{str(item['profile_list'][i][0]):^{max_shape_width}}|"
            info += f"{str(item['profile_list'][i][1]):^{max_shape_width}}|"
            info += f"{str(item['profile_list'][i][2]):^{max_shape_width}}|"
            print(info)

    return
