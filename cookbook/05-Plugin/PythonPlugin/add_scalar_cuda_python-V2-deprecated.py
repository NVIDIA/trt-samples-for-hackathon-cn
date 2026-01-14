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

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
#from polygraphy.json import from_json, to_json  # use this if ndarray need to be serialized
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from cuda.bindings import nvrtc
from tensorrt_cookbook import TRTWrapperV1, ceil_divide, check_array

scalar = 1.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar}

def checkNvrtcErrors(result):
    # Taken from https://github.com/NVIDIA/cuda-python/blob/main/examples/common/helper_cuda.py
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

# Use `class KernelHelper` from https://github.com/NVIDIA/cuda-python/blob/main/examples/common/common.py
def get_kernel(code, device_id, function_name):
    program = checkNvrtcErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, [], []))
    use_cubin = checkNvrtcErrors(nvrtc.nvrtcVersion())[-1] >= 1
    major = checkNvrtcErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id))
    minor = checkNvrtcErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id))
    arch = bytes(f"--gpu-architecture={'sm' if use_cubin else 'compute'}_{major}{minor}", "ascii")
    include_dir = os.path.join(os.getenv("CUDA_HOME"), "include")
    opts = [b"--fmad=true", arch, "--include-path={}".format(include_dir).encode("UTF-8"), b"--std=c++11", b"-default-device"]
    try:
        checkNvrtcErrors(nvrtc.nvrtcCompileProgram(program, len(opts), opts))
    except RuntimeError as err:
        logSize = checkNvrtcErrors(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * logSize
        checkNvrtcErrors(nvrtc.nvrtcGetProgramLog(program, log))
        print(log.decode())
        print(err)
        exit(-1)
    if use_cubin:
        data_size = checkNvrtcErrors(nvrtc.nvrtcGetCUBINSize(program))
        data = b" " * data_size
        checkNvrtcErrors(nvrtc.nvrtcGetCUBIN(program, data))
    else:
        data_size = checkNvrtcErrors(nvrtc.nvrtcGetPTXSize(program))
        data = b" " * data_size
        checkNvrtcErrors(nvrtc.nvrtcGetPTX(program, data))
    module = checkNvrtcErrors(cuda.cuModuleLoadData(np.char.array(data)))
    return checkNvrtcErrors(cuda.cuModuleGetFunction(module, function_name))

source_code = r"""
#include <cuda_fp16.h>
#define F(T)                                                                        \
extern "C" __global__                                                               \
void addScalarKernel_##T(T const* x, T* y, float const scalar, int const nElement)  \
{                                                                                   \
    const int index = blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (index >= nElement)                                                          \
        return;                                                                     \
    float _1 = (float)x[index];                                                     \
    float _2 = _1 + scalar;                                                         \
    y[index] = (T)_2;                                                               \
}

F(float)
F(half)
"""

class AddScalarPlugin(trt.IPluginV2DynamicExt):

    def __init__(self, scalar):
        trt.IPluginV2DynamicExt.__init__(self)
        self.plugin_type = "AddScalar"  # necessary as function `getPluginType` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.num_outputs = 1  # necessary as function `getNbOutputs` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.scalar = scalar  # metadata of the plugin
        self.device = 0  # default device is cuda:0, can be get by `cuda.cuDeviceGet(0)`
        #print(self.tensorrt_version)  # a read-only attribution noting API version with which this plugin was built
        #print(self.serialization_size)  # a read-only attribution noting the size of the serialization buffer required
        return

    def initialize(self) -> int:
        return 0

    def terminate(self) -> None:
        return

    def get_serialization_size(self) -> int:
        m = {"scalar": self.scalar}
        return len(json.dumps(m))
        #return len(to_json(m))  # use this if m contains ndarray

    def serialize(self) -> bytes:
        m = {"scalar": self.scalar}
        return json.dumps(m)
        #return to_json(m)  # use this if m contains ndarray

    def destroy(self) -> None:
        return

    def get_output_datatype(self, output_index: int, input_types: List[trt.DataType]) -> trt.DataType:
        if output_index == 0:
            return input_types[0]
        else:
            return None

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned_plugin = AddScalarPlugin(0.0)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        if output_index == 0:
            output_dims = trt.DimsExprs(inputs[0])
            for i in range(len(inputs[0])):
                output_dims[i] = inputs[0][i]
            return output_dims
        else:
            return None

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        res = False
        desc = in_out[pos]
        if pos == 0:
            res = (desc.type == trt.float32 or desc.type == trt.float16) and desc.format == trt.TensorFormat.LINEAR
        elif pos == 1:
            res = desc.type == in_out[0].type and desc.format == trt.TensorFormat.LINEAR
        if False:  # Print information about the input / output
            info = f"    {pos=}:"
            info += f"{[str(in_out[i].type)[9:] for i in range(len(in_out))]},"
            info += f"{[str(in_out[i].format)[13:] for i in range(len(in_out))]}"
            info += f"->{res=}"
            print(info)
        return res

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_workspace_size(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        n_element = np.prod(np.array(input_desc[0].dims))

        kernel_name = b"addScalarKernel_half" if input_desc[0].type == trt.float16 else b"addScalarKernel_float"
        kernel = get_kernel(source_code, self.device, kernel_name)
        block_size = 256
        grid_size = ceil_divide(n_element, block_size)
        p_stream = np.array([stream], dtype=np.uint64)

        p_input = np.array([inputs[0]], dtype=np.uint64)
        p_output = np.array([outputs[0]], dtype=np.uint64)
        p_scalar = np.array(self.scalar, dtype=np.float32)
        p_element = np.array(n_element, dtype=np.int32)
        args = np.array([arg.ctypes.data for arg in [p_input, p_output, p_scalar, p_element]], dtype=np.uint64)

        checkNvrtcErrors(cuda.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, p_stream, args, 0))
        return

class AddScalarPluginCreator(trt.IPluginCreator):

    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "AddScalar"  # necessary as function `getPluginName` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.field_names = trt.PluginFieldCollection([trt.PluginField("scalar", np.array([]), trt.PluginFieldType.FLOAT32)])
        return

    def create_plugin(self, name, plugin_field_collection):
        scalar = 0.0
        for plugin_field in plugin_field_collection:
            if plugin_field.name == "scalar":
                scalar = float(plugin_field.data[0])
        return AddScalarPlugin(scalar)

    def deserialize_plugin(self, name, data):
        deserialized = AddScalarPlugin(0.0)
        m = dict(json.loads(data))
        #m = dict(from_json(data))  # use this if m contains ndarray
        deserialized.__dict__.update(m)
        return deserialized

def test_case(b_FP16):
    if b_FP16:
        input_data["inputT0"] = input_data["inputT0"].astype(np.float16)
        trt_file = Path(f"model-fp16.trt")
        trt_datatype = trt.float16
    else:
        trt_file = Path(f"model-fp32.trt")
        trt_datatype = trt.float32

    tw = TRTWrapperV1(trt_file=trt_file)
    if tw.engine_bytes is None:  # need to create engine from scratch
        if b_FP16:
            tw.config.set_flag(trt.BuilderFlag.FP16)
        plugin_creator = trt.get_plugin_registry().get_plugin_creator("AddScalar", "1", "")
        field_list = [trt.PluginField("scalar", np.array(1.0, dtype=np.float32), trt.PluginFieldType.FLOAT32)]
        field_collection = trt.PluginFieldCollection(field_list)
        plugin = plugin_creator.create_plugin("AddScalar", field_collection)

        input_tensor = tw.network.add_input("inputT0", trt_datatype, [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v2([input_tensor], plugin)
        layer.precision = trt_datatype
        tensor = layer.get_output(0)
        tensor.dtype = trt_datatype
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    tw.setup(input_data)
    tw.infer(b_print_io=False)

    output_cpu = add_scalar_cpu(input_data, scalar)

    check_array(tw.buffer["outputT0"][0], output_cpu["outputT0"], True)

if __name__ == "__main__":
    os.system("rm -rf ./*.trt")

    # We should only operate the resources once cross sessions
    plugin_registry = trt.get_plugin_registry()
    my_plugin_creator = AddScalarPluginCreator()
    if my_plugin_creator.name not in [creator.name for creator in plugin_registry.all_creators]:
        plugin_registry.register_creator(my_plugin_creator, "")

    test_case(False)  # Build engine and plugin to do inference
    test_case(False)  # Load engine and plugin to do inference
    test_case(True)
    test_case(True)

    print("Finish")
