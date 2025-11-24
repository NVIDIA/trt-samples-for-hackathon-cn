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

from tensorrt_cookbook import APIExcludeSet, TRTWrapperV1

tw = TRTWrapperV1()
config = tw.config

callback_member, callable_member, attribution_member = APIExcludeSet.split_members(config)
print(f"\n{'='*64} Members of trt.IBuilderConfig:")
print(f"{len(callback_member):2d} Members to get/set callback classes: {callback_member}")
print(f"{len(callable_member):2d} Callable methods: {callable_member}")
print(f"{len(attribution_member):2d} Non-callable attributions: {attribution_member}")

config.reset()  # Reset BuidlerConfig to default

print(f"{config.progress_monitor = }")  # 04-Feature/ProgressMonitor, get/set progress_monitor

input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(tw.profile)

layer = tw.network.add_identity(input_tensor)
tw.network.mark_output(layer.get_output(0))
tw.builder.build_serialized_network(tw.network, config)
tw.builder.build_engine_with_config(tw.network, config)

print(f"\n{'='*64} Device related")
print(f"{config.engine_capability = }")
# Alternative values of trt.EngineCapability:
# trt.EngineCapability.STANDARD         -> 0, default without targeting safety runtime, supporting GPU and DLA
# trt.EngineCapability.SAFETY           -> 1, targeting safety runtime, supporting GPU on NVIDIA Drive(R) products
# trt.EngineCapability.DLA_STANDALONE   -> 2, targeting DLA runtime, supporting DLA
print(f"{config.runtime_platform = }")
print(f"{config.tiling_optimization_level = }")
print(f"{config.l2_limit_for_tiling = }")
print(f"{config.default_device_type = }")
print(f"{config.DLA_core = }")

print(f"{config.can_run_on_DLA(layer) = }")
config.set_device_type(layer, trt.DeviceType.DLA)  # Set the device on which one layer run
# Alternative values of trt.DeviceType:
# trt.DeviceType.GPU    -> 0
# trt.DeviceType.DLA    -> 1
print(f"{config.get_device_type(layer) = }")
print(f"{config.is_device_type_set(layer) = }")
config.reset_device_type(layer)

print(f"\n{'='*64} Flag related")

def print_flag():
    flags = []
    for name, (value, text) in trt.BuilderFlag.__entries.items():
        flags.append([value, name])
    flags = sorted(flags, key=lambda x: x[0])
    for value, name in flags:
        print(f"{int(value):3d}: {name}")
    return

print_flag()  # print all flags

print(f"{config.flags = }")  # Get/set flags, TF32 (1<<6) is set as default on Ampere above GPU
config.set_flag(trt.BuilderFlag.DEBUG)  # Set single flag
config.set_flag(trt.BuilderFlag.DEBUG)  # Get single flag
config.clear_flag(trt.BuilderFlag.DEBUG)  # Unset single flag
config.flags = 1 << int(trt.BuilderFlag.DEBUG) | 1 << int(trt.BuilderFlag.REFIT)  # Set multiple flags
config.flags = 0  # unset all flags

print(f"{config.quantization_flags = }")  # Get/set quantization flags, 0 as default
print(f"{config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) = }")  # get whether the quantization flag is enabled
config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # set single quantization flag
config.clear_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # clear single quantization flag

print(f"\n{'='*64} Preview feature related")
print(f"{config.get_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806)}")  # check whether the preview feature is enabled
config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
# Alternative values of trt.PreviewFeature:
# trt.PreviewFeature.PROFILE_SHARING_0806               -> 0
# trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03            -> 1
# trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10    -> 2

print(f"\n{'='*64} Engine related")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = } Bytes")  # all GPU memory is used by default
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM) = } Bytes")
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# Alternative values of trt.MemoryPoolType:
# trt.MemoryPoolType.WORKSPACE              -> 0
# trt.MemoryPoolType.DLA_MANAGED_SRAM       -> 1
# trt.MemoryPoolType.DLA_LOCAL_DRAM         -> 2
# trt.MemoryPoolType.DLA_GLOBAL_DRAM        -> 3
# trt.MemoryPoolType.TACTIC_DRAM            -> 4
# trt.MemoryPoolType.TACTIC_SHARED_MEMORY   -> 5

print(f"{config.num_optimization_profiles = }")  # Get number of Optimization-Profile, default value: 0
print(f"{config.max_num_tactics = }")  # Get maximum count of tactic to try during building
print(f"{config.builder_optimization_level = }")  # Get/set optimization level, default value: 3
print(f"{config.profile_stream = }")  # Get/set the CUDA stream for auto tuning, default value: 0
print(f"{config.avg_timing_iterations = }")  # Get/set average times to running each tactic during auto tuning, default value: 1
print(f"{config.hardware_compatibility_level = }")  # Get/set hardware compatibility level, default value: trt.HardwareCompatibilityLevel.NONE
# Alternative values of trt.HardwareCompatibilityLevel:
# trt.HardwareCompatibilityLevel.NONE                       -> 0
# trt.HardwareCompatibilityLevel.AMPERE_PLUS                -> 1
# trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY    -> 2

print(f"{config.max_aux_streams = }")  # Get/set auxiliary CUDA streams to do inference, default value: -1
print(f"{config.plugins_to_serialize = }")

print(f"{config.get_tactic_sources() = }")  # get tactic sources, default value: 24
config.set_tactic_sources(0)
# Alternative values as argument (bit mask)
# trt.TacticSource.CUBLAS                   -> 0, deprecated
# trt.TacticSource.CUBLAS_LT                -> 1, deprecated
# trt.TacticSource.CUDNN                    -> 2, deprecated
# trt.TacticSource.EDGE_MASK_CONVOLUTIONS   -> 3
# trt.TacticSource.JIT_CONVOLUTIONS         -> 4

print(f"{config.profiling_verbosity = }")  # Get/set profiling verbosity
# Alternative values of trt.ProfilingVerbosity:
# trt.ProfilingVerbosity.LAYER_NAMES_ONLY   -> 0, default
# trt.ProfilingVerbosity.NONE               -> 1
# trt.ProfilingVerbosity.DETAILED           -> 2

print("Finish")
"""
APIs not showed:
algorithm_selector      -> deprecated, 04-Feature/AlgorithmSelector
create_timing_cache     -> 04-Feature/TimingCache
get_calibration_profile -> deprecated
get_timing_cache        -> 04-Feature/TimingCache
int8_calibrator         -> deprecated
set_calibration_profile -> deprecated
set_timing_cache        -> 04-Feature/TimingCache
"""
