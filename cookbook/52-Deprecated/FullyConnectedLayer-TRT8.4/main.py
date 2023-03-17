#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # Deprecated in TensorRT 8.4
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # use this since TensorRT 8.0

inputTensor = network.add_input("inputT0", trt.float32, [-1, 32, 1, 1])
profile.set_shape(inputTensor.name, [1, 32, 1, 1], [4, 32, 1, 1], [16, 32, 1, 1])
config.add_optimization_profile(profile)

weight = trt.Weights(np.ones([32, 64], dtype=np.float32))
bias = trt.Weights(np.ones([64], dtype=np.float32))
identityLayer = network.add_fully_connected(inputTensor, 64, weight, bias)
network.mark_output(identityLayer.get_output(0))

engineString = builder.build_serialized_network(network, config)
