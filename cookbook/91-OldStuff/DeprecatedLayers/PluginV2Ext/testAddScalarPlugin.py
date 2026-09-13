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

import ctypes
import os

import numpy as np
import tensorrt as trt
from cuda import cudart
from tensorrt_cookbook import case_mark, check_array

plugin_file = "./AddScalarPlugin.so"

def add_scalar_cpu(buffer, scalar):
    return [buffer[0] + scalar]

def get_add_scalar_plugin(scalar):
    # Deprecated interface: IPluginV2Ext plugins are loaded through the removed add_plugin_v2 API; use IPluginV3 + add_plugin_v3 instead.
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == "AddScalar":
            parameter_list = [trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32)]
            return creator.create_plugin(creator.name, trt.PluginFieldCollection(parameter_list))
    return None

@case_mark
def run(shape, scalar):
    # This PluginV2Ext example relies on implicit-batch mode, which TRTWrapperV1 (explicit batch) cannot express, so minimal manual boilerplate is kept.
    trt_file = "./model-Dim%s.trt" % str(len(shape))
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    ctypes.cdll.LoadLibrary(plugin_file)
    if os.path.isfile(trt_file):
        with open(trt_file, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            print("Fail loading engine")
            return
        print("Succeed loading engine")
    else:
        builder = trt.Builder(logger)
        builder.max_batch_size = 32
        network = builder.create_network()  # implicit-batch mode is mandatory for add_plugin_v2
        builder_config = builder.create_builder_config()

        tensor = network.add_input("inputT0", trt.float32, shape[1:])
        layer = network.add_plugin_v2([tensor], get_add_scalar_plugin(scalar))
        network.mark_output(layer.get_output(0))
        engine_string = builder.build_serialized_network(network, builder_config)
        if engine_string is None:
            print("Fail building engine")
            return
        print("Succeed building engine")
        with open(trt_file, "wb") as f:
            f.write(engine_string)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

    n_io = engine.num_io_tensors
    tensor_name_list = [engine.get_tensor_name(i) for i in range(n_io)]
    n_input = [engine.get_tensor_mode(name) for name in tensor_name_list].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()

    buffer_h = []
    for i in range(n_input, n_io):
        buffer_h.append(np.empty((shape[0], ) + tuple(context.get_tensor_shape(tensor_name_list[i])), dtype=trt.nptype(engine.get_tensor_dtype(tensor_name_list[i]))))
    buffer_d = []
    for i in range(n_io):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])

    buffer_h[0] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    for i in range(n_input):
        cudart.cudaMemcpy(buffer_d[i], buffer_h[i].ctypes.data, buffer_h[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(n_io):
        context.set_tensor_address(tensor_name_list[i], int(buffer_d[i]))

    context.execute(shape[0], buffer_d)

    for i in range(n_input, n_io):
        cudart.cudaMemcpy(buffer_h[i].ctypes.data, buffer_d[i], buffer_h[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    output_cpu = add_scalar_cpu(buffer_h[:n_input], scalar)
    check_array(buffer_h[n_input:][0], output_cpu[0], True)

    for b in buffer_d:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.trt")

    # Build engine and plugin to do inference
    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)
    # Load engine and plugin to do inference
    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)

    print("Finish")
