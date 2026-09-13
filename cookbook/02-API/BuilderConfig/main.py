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

import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, check_api_coverage, print_enumerated_members

tw = TRTWrapperV1()
builder_config = tw.builder_config

check_api_coverage(builder_config)  # Sanity check, unnecessary in normal workflow

print(f"\n{'=' * 64} Usage show")

builder_config.reset()  # Reset BuilderConfig to default

print(f"{builder_config.progress_monitor = }")  # Get/set progress_monitor, 04-Feature/ProgressMonitor

# Build a network to use other APIs
input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
builder_config.add_optimization_profile(tw.profile)

layer = tw.network.add_identity(input_tensor)
tw.network.mark_output(layer.get_output(0))
tw.builder.build_serialized_network(tw.network, builder_config)
tw.builder.build_engine_with_config(tw.network, builder_config)

print(f"\n{'-' * 64} Device related")
print_enumerated_members(trt.EngineCapability)
print(f"{builder_config.engine_capability = }")
print(f"{builder_config.runtime_platform = }")
print(f"{builder_config.tiling_optimization_level = }")
print(f"{builder_config.l2_limit_for_tiling = }")
print(f"{builder_config.default_device_type = }")
print(f"{builder_config.DLA_core = }")

print(f"{builder_config.can_run_on_DLA(layer) = }")
builder_config.set_device_type(layer, trt.DeviceType.DLA)  # Set the device on which one layer run
# Alternative values of trt.DeviceType:
# trt.DeviceType.GPU    -> 0
# trt.DeviceType.DLA    -> 1
print(f"{builder_config.get_device_type(layer) = }")
print(f"{builder_config.is_device_type_set(layer) = }")
builder_config.reset_device_type(layer)

print(f"\n{'-' * 64} trt.BuilderFlag related")

print(f"{builder_config.flags = }")  # Get/set flags, TF32 (1<<6) is set as default on Ampere above GPU
builder_config.set_flag(trt.BuilderFlag.DEBUG)  # Set single flag
builder_config.get_flag(trt.BuilderFlag.DEBUG)  # Get single flag
builder_config.clear_flag(trt.BuilderFlag.DEBUG)  # Unset single flag
builder_config.flags = 1 << int(trt.BuilderFlag.DEBUG) | 1 << int(trt.BuilderFlag.REFIT)  # Set multiple flags
builder_config.flags = 0  # unset all flags

print_enumerated_members(trt.BuilderFlag)
# This file only shows the *shape* of the flag API, with DEBUG as a stand-in; it never builds with
# any flag, so it cannot say what one does. For all 20 flags measured one at a time against the same
# network -- plus a map of where each is really demonstrated -- see `02-API/BuilderFlag`.
# Two things worth knowing before you use the API above:
#   * TF32 is ON by default (`flags == 64`), so assigning `flags` wholesale silently clears it.
#   * A flag being accepted says nothing about it having an effect; 14 of the 20 leave the plan
#     byte-identical because they act at runtime, in the build log, or on absent hardware.

print(f"\n{'-' * 64} trt.TilingOptimizationLevel related")
print_enumerated_members(trt.TilingOptimizationLevel)

builder_config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL  # Set the tiling optimization level
print(f"{builder_config.tiling_optimization_level = }")
builder_config.tiling_optimization_level = trt.TilingOptimizationLevel.NONE  # Restore default

print(f"\n{'=' * 64} Preview feature related")
print_enumerated_members(trt.PreviewFeature)
print(f"{builder_config.get_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03) = }")  # check whether the preview feature is enabled
builder_config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

print(f"\n{'-' * 64} Engine related")
print_enumerated_members(trt.MemoryPoolType)
for pool_type in trt.MemoryPoolType.__members__.values():
    size = builder_config.get_memory_pool_limit(pool_type)
    print(f"{pool_type = }, {size = } Bytes")
builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

print(f"{builder_config.num_optimization_profiles = }")  # Get number of Optimization-Profile, default: 0
print(f"{builder_config.max_num_tactics = }")  # Get maximum count of tactic to try during building
print(f"{builder_config.builder_optimization_level = }")  # Get/set optimization level, default: 3
print(f"{builder_config.profile_stream = }")  # Get/set the CUDA stream for auto tuning, default: 0
print(f"{builder_config.avg_timing_iterations = }")  # Get/set average times to running each tactic during auto tuning, default: 1

print_enumerated_members(trt.HardwareCompatibilityLevel)
print(f"{builder_config.hardware_compatibility_level = }")  # Get/set hardware compatibility level, default: trt.HardwareCompatibilityLevel.NONE

print(f"{builder_config.max_aux_streams = }")  # Get/set auxiliary CUDA streams to do inference, default: -1
print(f"{builder_config.plugins_to_serialize = }")

print_enumerated_members(trt.TacticSource)
print(f"{builder_config.get_tactic_sources() = }")  # get tactic sources, default: 24
builder_config.set_tactic_sources(0)

print_enumerated_members(trt.ProfilingVerbosity)
print(f"{builder_config.profiling_verbosity = }")  # Get/set profiling verbosity

print(f"{builder_config.remote_auto_tuning_config = }")  # Get/set remote auto-tuning configuration, default: ""
# This is only used for `builder_config.engine_capability = trt.EngineCapability.SAFETY`
# A example of remote auto-tuning configuration:
# "ssh://wili:wili@10.19.23.29:22?remote_exec_path=/usr/local/bin&remote_lib_path=/usr/lib/x86_64-linux-gnu&dump_remote_stdout=on&dump_remote_stderr=on"

timing_cache = tw.builder_config.create_timing_cache(b"")
tw.builder_config.set_timing_cache(timing_cache, False)  # Set timing cache, 04-Feature/TimingCache
tw.builder_config.get_timing_cache()  # Get timing cache, 04-Feature/TimingCache

print("Finish")
