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

import os
import numpy as np
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"

os.system("rm -rf ./*.plan")

logger = trt.Logger(trt.Logger.ERROR)

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # 配合 profiling_verbosity 以获得更多信息

inputTensor = network.add_input("inputT0", trt.float32, [-1, 1, 28, 28])
profile.set_shape(inputTensor.name, (1, 1, 28, 28), (4, 1, 28, 28), (8, 1, 28, 28))
config.add_optimization_profile(profile)

w = np.random.rand(32, 1, 5, 5).astype(np.float32).reshape(-1) * 2 - 1
b = np.random.rand(32).astype(np.float32).reshape(-1) * 2 - 1
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.random.rand(64, 32, 5, 5).astype(np.float32).reshape(-1) * 2 - 1
b = np.random.rand(64).astype(np.float32).reshape(-1) * 2 - 1
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.first_transpose = (0, 2, 3, 1)
_6.reshape_dims = (-1, 64 * 7 * 7, 1, 1)

w = np.random.rand(64 * 7 * 7, 1024).astype(np.float32).reshape(-1) * 2 - 1
b = np.random.rand(1024).astype(np.float32).reshape(-1) * 2 - 1
_7 = network.add_fully_connected(_6.get_output(0), 1024, w, b)
_8 = network.add_activation(_7.get_output(0), trt.ActivationType.RELU)

w = np.random.rand(1024, 10).astype(np.float32).reshape(-1) * 2 - 1
b = np.random.rand(10).astype(np.float32).reshape(-1) * 2 - 1
_9 = network.add_fully_connected(_8.get_output(0), 10, w, b)
_10 = network.add_activation(_9.get_output(0), trt.ActivationType.RELU)

_11 = network.add_shuffle(_10.get_output(0))
_11.reshape_dims = [-1, 10]

_12 = network.add_softmax(_11.get_output(0))
_12.axes = 1 << 1

_13 = network.add_topk(_12.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

network.mark_output(_13.get_output(1))

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

inspector = engine.create_engine_inspector()

print("Engine information:")  # engine information 等价于所有 layer information 放一起
print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))  # ONELINE or JSON

print("Layer information:")
for i in range(engine.num_layers):
    print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
