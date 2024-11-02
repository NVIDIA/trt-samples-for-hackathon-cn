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
from pathlib import Path

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1
from collections import OrderedDict
from cuda import cudart

shape = [3, 4, 5]

host_data = np.arange(10, dtype=np.float32)  # Host data we want to use in plugin (can be anything like complex structure or even nullptr)
p_host_data = np.array([host_data.ctypes.data], dtype=np.int64)  # Wrap pointer of host data into a numpy array
input_data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape), "inputT1": p_host_data}
trt_file = Path("model.trt")
plugin_file_list = [Path(__file__).parent / "PassHostDataPlugin.so"]

def getPassHostDataPlugin():
    name = "PassHostData"
    plugin_creator = trt.get_plugin_registry().get_creator(name, "1", "")
    if plugin_creator is None:
        print(f"Fail loading plugin {name}")
        return None
    field_list = []
    field_collection = trt.PluginFieldCollection(field_list)
    return plugin_creator.create_plugin(name, field_collection, trt.TensorRTPhase.BUILD)

def run():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        input_tensor0 = tw.network.add_input("inputT0", trt.float32, shape)  # Normal input tensor for the network
        input_tensor1 = tw.network.add_input("inputT1", trt.int64, [1])  # a pointer for host data

        layer = tw.network.add_plugin_v3([input_tensor0, input_tensor1], [], getPassHostDataPlugin())
        tensor = layer.get_output(0)
        tensor.name = "outputT0"

        tw.build([tensor])
        tw.serialize_engine(trt_file)

    # We need to bind the pointers of the data directly to TRT, so we do not use `tw.setup` or `tw.infer` here
    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
    tw.context = tw.engine.create_execution_context()

    tw.tensor_name_list = [tw.engine.get_tensor_name(i) for i in range(tw.engine.num_io_tensors)]
    tw.n_input = sum([tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in tw.tensor_name_list])
    tw.n_output = tw.engine.num_io_tensors - tw.n_input

    for name, data in input_data.items():
        tw.context.set_input_shape(name, data.shape)

    tw.buffer = OrderedDict()
    for name in tw.tensor_name_list:
        data_type = tw.engine.get_tensor_dtype(name)
        runtime_shape = tw.context.get_tensor_shape(name)
        n_byte = trt.volume(runtime_shape) * data_type.itemsize
        host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
        if name == "inputT1":
            device_buffer = None  # No need for device buffer for the host data
        else:
            device_buffer = cudart.cudaMalloc(n_byte)[1]
        tw.buffer[name] = [host_buffer, device_buffer, n_byte]

    for name, data in input_data.items():
        tw.buffer[name][0] = np.ascontiguousarray(data)

    for name in tw.tensor_name_list:
        if name == "inputT1":
            p_host_data = input_data["inputT1"]
            p_p_host_data = p_host_data.ctypes.data
            print(f"[Python]pData  = 0x{p_host_data[0]:x}")
            print(f"[Python]ppData = 0x{p_p_host_data:x}")
            tw.context.set_tensor_address(name, p_p_host_data)  # bind host pointer directly for the host data
        else:
            tw.context.set_tensor_address(name, tw.buffer[name][1])  # USe device pointer for normal tensors

    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name == "inputT1":
                pass  # No need to copy the pointer of host data
            else:
                cudart.cudaMemcpy(tw.buffer[name][1], tw.buffer[name][0].ctypes.data, tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    tw.context.execute_async_v3(0)

    for name in tw.tensor_name_list:
        if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cudart.cudaMemcpy(tw.buffer[name][0].ctypes.data, tw.buffer[name][1], tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for _, device_buffer, _ in tw.buffer.values():
        cudart.cudaFree(device_buffer)

if __name__ == "__main__":
    os.system("rm -rf *.trt")

    run()  # Build engine and plugin to do inference
    run()  # Load engine and plugin to do inference

    print("Finish")
