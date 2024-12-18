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

import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)

builder = trt.Builder(logger)
builder.reset()  # reset Builder as default

print(f"{builder.logger is logger = }")  # get logger from builder

network = builder.create_network()
# Alternative values of trt.NetworkDefinitionCreationFlag:
# trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH  -> 0, use Explicit-Batch mode (default)
# trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED  -> 1, use Strong-Typed mode
# Input argument of create_network() is a bit mask of trt.NetworkDefinitionCreationFlag, and 0 (default) and 1 has the same meaning.

config = builder.create_builder_config()
profile = builder.create_optimization_profile()
inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

print(f"\n{'='*64} Device related")
print(f"{builder.platform_has_tf32 = }")
print(f"{builder.platform_has_fast_fp16 = }")
print(f"{builder.platform_has_fast_int8 = }")
print(f"{builder.num_DLA_cores = }")
print(f"{builder.max_DLA_batch_size = }")

print("\n{'='*64} Engine related")
builder.max_threads = 16  # set the maximum threads used during buildtime, unreadable
print(f"{builder.is_network_supported(network, config) = }")  # whether the network is fully supported by TensorRT
print(f"{builder.get_plugin_registry() = }")
print(f"{builder.error_recorder = }")  # -> 04-Feature/ErrorRecorder
#print(f"{builder.gpu_allocator = }")  # only can be set

engineString = builder.build_serialized_network(network, config)

print("Finish")
