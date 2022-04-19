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

def build_engine(shape_data, shape_indices, shape_updates):
    plugin_creator = get_plugin_creator('ScatterND')
    if plugin_creator  == None:
        print('scatterND plugin not found. Exiting')
        exit()

    builder = trt.Builder(logger)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    tensor_data = network.add_input('data', trt.DataType.FLOAT, shape_data)
    tensor_indices = network.add_input('indices', trt.DataType.INT32, shape_indices)
    tensor_updates = network.add_input('updates', trt.DataType.FLOAT, shape_updates)

    layer  = network.add_plugin_v2(
            [tensor_data, tensor_indices, tensor_updates], 
            plugin_creator .create_plugin('ScatterND', trt.PluginFieldCollection([
            ]))
        )
    network.mark_output(layer.get_output(0))

    return builder.build_engine(network, builder.create_builder_config())

def run_trt(data, indices, updates, output_0):

    engine = build_engine(data.shape, indices.shape, updates.shape)
    print("succeed to build the engine")

    context = engine.create_execution_context()
    
    d_indices = cuda.mem_alloc(indices.nbytes)
    d_data = cuda.mem_alloc(data.nbytes)
    d_updates = cuda.mem_alloc(updates.nbytes)

    d_output_0 = cuda.mem_alloc(output_0.nbytes)

    cuda.memcpy_htod(d_indices, indices)
    cuda.memcpy_htod(d_data, data)
    cuda.memcpy_htod(d_updates, updates)
    bindings = [int(d_data), int(d_indices), int(d_updates), int(d_output_0)]    
    
    context.execute_v2(bindings)

    cuda.memcpy_dtoh(output_0, d_output_0)
    
    return output_0

if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.INFO)
    ctypes.cdll.LoadLibrary('../build/ScatterND.so')

    data = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
    indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array(
        [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
    # Expecting output as np.array(
    #    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
    #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
    #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
    
    output_0 = np.zeros(data.shape, dtype=np.float32)

    output_0 = run_trt(data, indices, updates, output_0)

    print(f"data shape: {data.shape} indices shape: {indices.shape} updates shape: {updates.shape} output shape: {output_0.shape} ")
    print(f"data {data} \n indices: \n {indices} \n updates: \n {updates} \n result: \n {output_0}")
