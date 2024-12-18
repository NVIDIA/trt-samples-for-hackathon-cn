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

import numpy as np
import tensorrt as trt

shape = [1, 3, 4, 5]
input_data = {}
input_data["inputT0"] = np.zeros(np.prod(shape), dtype=np.float32).reshape(shape)  # Execution input tensor
input_data["inputT1"] = np.array(shape, dtype=np.int32)  # Shape input tensor

logger = trt.Logger(trt.Logger.Severity.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

tensor0 = network.add_input("inputT0", trt.float32, [-1 for _ in shape])
tensor1 = network.add_input("inputT1", trt.int32, [len(shape)])
profile.set_shape(tensor0.name, [1 for _ in shape], shape, shape)
profile.set_shape_input(tensor1.name, [1 for _ in shape], shape, shape)
config.add_optimization_profile(profile)

kernel = np.ascontiguousarray(np.ones([1, 3, 3, 3], dtype=np.float32))
bias = np.ascontiguousarray(np.ones([1], dtype=np.float32))
layer0 = network.add_convolution_nd(tensor0, 1, [3, 3], kernel, bias)
tensor0 = layer0.get_output(0)
tensor0.name = "outputT0"
layer1 = network.add_identity(tensor1)
tensor1 = layer1.get_output(0)
tensor1.name = "outputT1"

network.set_weights_name(trt.Weights(kernel), "kernel of conv")  # set name of weight in network level
#network.remove_tensor(tensor0)  # can be only used in ONNX workflow

network.mark_debug(tensor0)  # -> 04-Feature/DebugTensor
network.is_debug_tensor(tensor0)
network.unmark_debug(tensor0)

network.mark_output(tensor0)

network.mark_output(tensor1)
network.unmark_output(tensor1)
network.mark_output_for_shapes(tensor1)
network.unmark_output_for_shapes(tensor1)

print(f"{network.name = }")
print(f"{network.builder = }")
print(f"{network.error_recorder = }")  # -> 04-Feature/ErrorRecorder
#print(f"{network.has_implicit_batch_dimension = }")  # deprecated API

print(f"{network.flags = }")
print(f"{network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) = }")

# A simplified version of 07-Tool/NetworkPrinter
print(f"{network.num_inputs = }")
for i in range(network.num_inputs):
    print(f"    network.get_input({i}) = {network.get_input(i)}")

print(f"{network.num_outputs = }")
for i in range(network.num_outputs):
    print(f"    network.get_output({i}) = {network.get_output(i)}")

print(f"{network.num_layers = }")
for i in range(network.num_layers):
    print(f"    network.get_layer({i}) = {network.get_layer(i)}")

print("Finish")
"""
APIs not showed:
add_*         -> 02-API/Layer
"""
