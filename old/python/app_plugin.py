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

from functools import reduce
import numpy as np
import tensorrt
import ctypes
import torch
from trt_lite2 import TrtLite

np.set_printoptions(threshold=np.inf)

ctypes.cdll.LoadLibrary('../build/AddPlugin.so')

def get_plugin_creator(plugin_name):
    plugin_creator_list = tensorrt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(builder, input_shape):
    plugin_creator = get_plugin_creator('AddPlugin')
    if plugin_creator == None:
        print('Plugin not found. Exiting')
        exit()

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20

    builder.max_batch_size = 8
    network = builder.create_network()
    tensor = network.add_input('data', tensorrt.DataType.FLOAT, input_shape)
    
    layer = network.add_plugin_v2(
        [tensor], 
        plugin_creator.create_plugin('AddPlugin', tensorrt.PluginFieldCollection([
            tensorrt.PluginField('valueToAdd', np.array([100.0], dtype=np.float32), tensorrt.PluginFieldType.FLOAT32)
        ]))
    )
    tensor = layer.get_output(0)
    network.mark_output(tensor)

    return builder.build_engine(network, config)

def run_engine():
    batch_size = 2
    input_shape = (batch_size, 1, 2, 8)
    n = reduce(lambda x, y: x * y, input_shape)
    input_data = np.asarray(range(n), dtype=np.float32).reshape(input_shape)
    output_data = np.zeros(input_shape, dtype=np.float32)
    
    trt = TrtLite(build_engine, (input_shape[1:],))
    trt.print_info()

    d_buffers = trt.allocate_io_buffers(batch_size, True)

    d_buffers[0].copy_(torch.from_numpy(input_data))
    trt.execute([t.data_ptr() for t in d_buffers], batch_size)
    output_data = d_buffers[1].cpu().numpy()
    
    print(output_data)

if __name__ == '__main__':
    run_engine()
