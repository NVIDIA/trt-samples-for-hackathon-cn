#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import torch
import triton
import triton.language as tl
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, ceil_divide, check_array, datatype_np_to_trt

scalar = 1.0
shape = [3, 4, 5]
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

def add_scalar_cpu(buffer, scalar):
    return {"outputT0": buffer["inputT0"] + scalar}

@triton.jit
def add_scalar(X, Y, scalar, n_element, BLOCK_SIZE: tl.constexpr):
    # tl.program_id(0) -> blockIdx.x
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_element
    x = tl.load(X + i, mask=mask)
    w = x + scalar
    tl.store(Y + i, w, mask=mask)

class AddScalarPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self, scalar: float):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "AddScalar"  # necessary as function `getPluginType` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.num_outputs = 1  # necessary as function `getNbOutputs` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.scalar = scalar  # metadata of the plugin
        self.device = 0  # default device is cuda:0, can be get by `cuda.cuDeviceGet(0)`
        return

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def clone(self) -> trt.IPluginV3:
        cloned_plugin = AddScalarPlugin(0.0)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0]]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        output_dims = trt.DimsExprs(inputs[0])
        return [output_dims]

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        res = False
        desc = in_out[pos].desc
        if pos == 0:
            res = (desc.type == trt.float32 or desc.type == trt.float16) and desc.format == trt.TensorFormat.LINEAR
        elif pos == 1:
            res = desc.type == in_out[0].desc.type and desc.format == trt.TensorFormat.LINEAR
        if False:  # print information about the input / output
            info = f"    {pos=}:"
            info += f"{[str(in_out[i].type)[9:] for i in range(len(in_out))]},"
            info += f"{[str(in_out[i].format)[13:] for i in range(len(in_out))]}"
            info += f"->{res=}"
            print(info)
        return res

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self: trt.IPluginV3, tactic: int) -> None:
        return None

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> ModuleNotFoundError:
        return None

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        data_type = trt.nptype(input_desc[0].type)
        shape = input_desc[0].dims
        n_element = np.prod(shape)
        buffer_size = n_element * np.dtype(data_type).itemsize

        c_data_type = ctypes.c_int16 if data_type == np.float16 else ctypes.c_float
        p_input = ctypes.cast(inputs[0], ctypes.POINTER(c_data_type * n_element))[0]
        p_input = np.ndarray(shape, dtype=data_type, buffer=p_input)
        p_input = torch.as_tensor(p_input, device='cuda')
        p_output = ctypes.cast(outputs[0], ctypes.POINTER(c_data_type * n_element))[0]
        p_output = np.ndarray(shape, dtype=data_type, buffer=p_output)
        p_output = torch.as_tensor(p_output, device='cuda')

        block_size = 256
        grid_size = ceil_divide(n_element, block_size)
        add_scalar[(grid_size, )](p_input, p_output, float(self.scalar), int(n_element), BLOCK_SIZE=block_size)
        cudart.cudaMemcpyAsync(outputs[0], p_output.data_ptr(), buffer_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
        """
        # Equivalent implementation, but need to use package `cupy`
        import cupy as cp
        data_type = trt.nptype(input_desc[0].type)
        n_element = np.prod(np.array(input_desc[0].dims))
        buffer_size = n_element * np.dtype(data_type).itemsize

        p_input = cp.cuda.UnownedMemory(inputs[0], buffer_size, self)
        p_input = cp.cuda.MemoryPointer(p_input, 0)
        p_input = cp.ndarray(n_element, dtype=data_type, memptr=p_input)
        p_input = torch.as_tensor(p_input, device='cuda')

        p_output = cp.cuda.UnownedMemory(outputs[0], buffer_size, self)
        p_output = cp.cuda.MemoryPointer(p_output, 0)
        p_output = cp.ndarray(n_element, dtype=data_type, memptr=p_output)
        p_output = torch.as_tensor(p_output, device='cuda')

        block_size = 256
        grid_size = ceil_divide(n_element, block_size)
        add_scalar[(grid_size, )](p_input, p_output, float(self.scalar), int(n_element), BLOCK_SIZE=block_size)
        """
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        field_collection = trt.PluginFieldCollection([trt.PluginField("scalar", np.array(self.scalar, dtype=np.float32), trt.PluginFieldType.FLOAT32)])
        return field_collection

class AddScalarPluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "AddScalar"  # necessary as function `getPluginName` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.field_names = trt.PluginFieldCollection([trt.PluginField("scalar", np.array([]), trt.PluginFieldType.FLOAT32)])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        scalar = 0.0
        for f in field_collection:
            if f.name == "scalar":
                scalar = float(f.data[0])
        return AddScalarPlugin(scalar)

def test_case(precision):
    trt_file = Path(f"model-{'fp32' if precision == np.float32 else 'fp16'}.trt")
    tw = TRTWrapperV1(trt_file=trt_file)
    if tw.engine_bytes is None:  # need to create engine from scratch
        if precision == np.float16:
            tw.config.set_flag(trt.BuilderFlag.FP16)
            input_data["inputT0"] = input_data["inputT0"].astype(np.float16)

        plugin_creator = trt.get_plugin_registry().get_creator("AddScalar", "1", "")
        field_list = [trt.PluginField("scalar", np.array(1.0, dtype=np.float32), trt.PluginFieldType.FLOAT32)]
        field_collection = trt.PluginFieldCollection(field_list)
        plugin = plugin_creator.create_plugin("AddScalar", field_collection, trt.TensorRTPhase.BUILD)

        input_tensor = tw.network.add_input("inputT0", datatype_np_to_trt(precision), [-1, -1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer = tw.network.add_plugin_v3([input_tensor], [], plugin)
        layer.precision = datatype_np_to_trt(precision)
        tensor = layer.get_output(0)
        tensor.dtype = datatype_np_to_trt(precision)
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

    test_case(np.float32)  # Build engine and plugin to do inference
    test_case(np.float32)  # Load engine and plugin to do inference
    test_case(np.float16)
    test_case(np.float16)

    print("Finish")
