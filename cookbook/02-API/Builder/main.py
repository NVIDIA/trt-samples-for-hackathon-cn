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

import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, CookbookStreamWriter, TRTWrapperV1

tw = TRTWrapperV1()
builder = tw.builder

callback_member, callable_member, attribution_member = APIExcludeSet.split_members(builder)
print(f"\n{'='* 64} Members of trt.IBuilder:")
print(f"{len(callback_member):2d} Members to get/set common/callback classes: {callback_member}")
print(f"{len(callable_member):2d} Callable methods: {callable_member}")
print(f"{len(attribution_member):2d} Non-callable attributions: {attribution_member}")

builder.reset()  # Reset Builder to default

print(f"{builder.error_recorder = }")  # Get/set error recorder, 04-Feature/ErrorRecorder
# print(f"{builder.gpu_allocator = }")  # Set gpu allocator, 04-Feature/GPUAllocator
print(f"{builder.logger = }")  # Get/set logger, 04-Feature/Logger

network = builder.create_network()
# Alternative argument:
# trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH              -> 0, use Explicit-Batch mode (default)
# trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED              -> 1, use Strong-Typed mode
# trt.NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS   -> 2, use Python JIT plugins
# # trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS -> 3, use Python AOT plugins

# Build a network to use other APIs
config = builder.create_builder_config()
profile = builder.create_optimization_profile()
input_tensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
identityLayer = network.add_identity(input_tensor)
network.mark_output(identityLayer.get_output(0))

print(f"\n{'=' * 64} Device related")
print(f"{builder.platform_has_tf32 = }")
print(f"{builder.platform_has_fast_fp16 = }")
print(f"{builder.platform_has_fast_int8 = }")
print(f"{builder.num_DLA_cores = }")
print(f"{builder.max_DLA_batch_size = }")

print(f"\n{'=' * 64} Engine related")
builder.max_threads = 16  # Set the maximum threads used for buildtime
print(f"{builder.is_network_supported(network, config) = }")  # Whether the network is fully supported by TensorRT
print(f"{builder.get_plugin_registry() = }")

engine = builder.build_engine_with_config(network, config)  # `trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)` is equivalent to `engine`
engine_bytes = builder.build_serialized_network(network, config)
engine_bytes = builder.build_serialized_network_to_stream(network, config, CookbookStreamWriter("engine.trt"))

print("Finish")
