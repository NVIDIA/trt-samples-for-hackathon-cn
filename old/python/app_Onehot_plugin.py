#!/usr/bin/python3

#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import ctypes

def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(shape_indices):
    plugin_creator = get_plugin_creator('OnehotPlugin')
    if plugin_creator == None:
        print('OnehotPlugin plugin not found. Exiting')
        exit()

    builder = trt.Builder(logger)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    tensor_indices = network.add_input('indices', trt.DataType.INT32, shape_indices)

    depth = 10
    layer = network.add_plugin_v2(
            [tensor_indices], plugin_creator.create_plugin('OnehotPlugin', trt.PluginFieldCollection([
            trt.PluginField('depth', np.array([depth], dtype=np.int32), trt.PluginFieldType.INT32)
            ])))
    network.mark_output(layer.get_output(0))

    return builder.build_engine(network, builder.create_builder_config())

def run_trt(indices, output_0):

    engine = build_engine(indices.shape)
    print("succeed to build the engine")
    context = engine.create_execution_context()

    d_indices = cuda.mem_alloc(indices.nbytes)
    d_output_0 = cuda.mem_alloc(output_0.nbytes)

    cuda.memcpy_htod(d_indices, indices)
    bindings = [int(d_indices), int(d_output_0)]
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output_0, d_output_0)

    return output_0

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.INFO)
    ctypes.cdll.LoadLibrary('../build/OnehotPlugin.so')

    indices = np.array([[1, 9], [2, 4]], dtype=np.int32)
    output_0 = np.zeros([2, 2, 10], dtype=np.float32)
    output_0 = run_trt(indices, output_0)

    print(f"indices shape: {indices.shape} output shape: {output_0.shape} ")
    print(f"indices: \n {indices} \n result: \n {output_0}")
