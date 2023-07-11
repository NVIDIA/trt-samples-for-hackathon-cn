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

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS  # turn on the switch of hardware compatibility, no other work needed

inputTensor = network.add_input("inputT0", trt.float32, [-1, 1024, 64])  # I write a "complex" network to see the performance differences between GPUs
profile.set_shape(inputTensor.name, [1, 1024, 64], [4, 1024,64], [16, 1024, 64])
config.add_optimization_profile(profile)

_0 = inputTensor
for i in range(64, 256):
    w = np.random.rand(1, i, i + 1).astype(np.float32)
    b = np.random.rand(1, 1, i + 1).astype(np.float32)
    _1 = network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
    _2 = network.add_matrix_multiply(_0, trt.MatrixOperation.NONE, _1.get_output(0), trt.MatrixOperation.NONE)
    _3 = network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
    _4 = network.add_elementwise(_2.get_output(0), _3.get_output(0), trt.ElementWiseOperation.SUM)
    _5 = network.add_activation(_4.get_output(0), trt.ActivationType.RELU)
    _0 = _5.get_output(0)

network.mark_output(_0)

engineString = builder.build_serialized_network(network, config)
with open("model.plan", "wb") as f:
    f.write(engineString)
    print("Succeeded saving .plan file!")

