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
from tensorrt_cookbook import TRTWrapperV1, check_api_coverage

tw = TRTWrapperV1()
builder_config = tw.builder_config

check_api_coverage(builder_config)  # Sanity check, unnecessary in normal workflow

print(f"\n{'=' * 64} Usage show")

builder_config.reset()  # Reset BuilderConfig to default

print(f"{builder_config.algorithm_selector = }")  # Get/set algorithm_selector, 04-Feature/ProgressMonitor, deprecated
print(f"{builder_config.int8_calibrator = }")  # Get/set int8_calibrator, 04-Feature/ProgressMonitor, deprecated
print(f"{builder_config.progress_monitor = }")  # Get/set progress_monitor, 04-Feature/ProgressMonitor

# Build a network to use other APIs
input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
tw.profile.set_shape(input_tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
builder_config.add_optimization_profile(tw.profile)

# Set /  get calibration profile for int8, deprecated
builder_config.set_calibration_profile(tw.profile)
builder_config.get_calibration_profile()

layer = tw.network.add_identity(input_tensor)
tw.network.mark_output(layer.get_output(0))
tw.builder.build_serialized_network(tw.network, builder_config)
tw.builder.build_engine_with_config(tw.network, builder_config)

print(f"\n{'-' * 64} Device related")
print(f"{builder_config.engine_capability = }")
# Alternative values of trt.EngineCapability:
# trt.EngineCapability.STANDARD         -> 0, default without targeting safety runtime, supporting GPU and DLA
# trt.EngineCapability.SAFETY           -> 1, targeting safety runtime, supporting GPU on NVIDIA Drive(R) products
# trt.EngineCapability.DLA_STANDALONE   -> 2, targeting DLA runtime, supporting DLA
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

def print_flag():
    flags = []
    for name, (value, text) in trt.BuilderFlag.__entries.items():
        flags.append([value, name])
    flags = sorted(flags, key=lambda x: x[0])
    for value, name in flags:
        print(f"{int(value):3d}: {name}")
    return

print_flag()  # print all flags

print(f"{builder_config.flags = }")  # Get/set flags, TF32 (1<<6) is set as default on Ampere above GPU
builder_config.set_flag(trt.BuilderFlag.DEBUG)  # Set single flag
builder_config.get_flag(trt.BuilderFlag.DEBUG)  # Get single flag
builder_config.clear_flag(trt.BuilderFlag.DEBUG)  # Unset single flag
builder_config.flags = 1 << int(trt.BuilderFlag.DEBUG) | 1 << int(trt.BuilderFlag.REFIT)  # Set multiple flags
builder_config.flags = 0  # unset all flags

print(f"{builder_config.quantization_flags = }")  # Get/set quantization flags, 0 as default
print(f"{builder_config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) = }")  # get whether the quantization flag is enabled
builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # set single quantization flag
builder_config.clear_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # clear single quantization flag

print(f"\n{'=' * 64} Preview feature related")
print(f"{builder_config.get_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806) = }")  # check whether the preview feature is enabled
builder_config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
# Alternative values of trt.PreviewFeature:
# trt.PreviewFeature.PROFILE_SHARING_0806               -> 0, deprecated
# trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03            -> 1
# trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10    -> 2
# trt.PreviewFeature.MULTIDEVICE_RUNTIME_10_16          -> 3

print(f"\n{'-' * 64} Engine related")
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = } Bytes")  # all GPU memory is used by default
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM) = } Bytes")
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM) = } Bytes")
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM) = } Bytes")
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) = } Bytes")
print(f"{builder_config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY) = } Bytes")
builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# Alternative values of trt.MemoryPoolType:
# trt.MemoryPoolType.WORKSPACE              -> 0
# trt.MemoryPoolType.DLA_MANAGED_SRAM       -> 1
# trt.MemoryPoolType.DLA_LOCAL_DRAM         -> 2
# trt.MemoryPoolType.DLA_GLOBAL_DRAM        -> 3
# trt.MemoryPoolType.TACTIC_DRAM            -> 4
# trt.MemoryPoolType.TACTIC_SHARED_MEMORY   -> 5

print(f"{builder_config.num_optimization_profiles = }")  # Get number of Optimization-Profile, default: 0
print(f"{builder_config.max_num_tactics = }")  # Get maximum count of tactic to try during building
print(f"{builder_config.builder_optimization_level = }")  # Get/set optimization level, default: 3
print(f"{builder_config.profile_stream = }")  # Get/set the CUDA stream for auto tuning, default: 0
print(f"{builder_config.avg_timing_iterations = }")  # Get/set average times to running each tactic during auto tuning, default: 1
print(f"{builder_config.hardware_compatibility_level = }")  # Get/set hardware compatibility level, default: trt.HardwareCompatibilityLevel.NONE
# Alternative values of trt.HardwareCompatibilityLevel:
# trt.HardwareCompatibilityLevel.NONE                       -> 0
# trt.HardwareCompatibilityLevel.AMPERE_PLUS                -> 1
# trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY    -> 2

print(f"{builder_config.max_aux_streams = }")  # Get/set auxiliary CUDA streams to do inference, default: -1
print(f"{builder_config.plugins_to_serialize = }")

print(f"{builder_config.get_tactic_sources() = }")  # get tactic sources, default: 24
builder_config.set_tactic_sources(0)
# Alternative argument (bit mask)
# trt.TacticSource.CUBLAS                   -> 0, deprecated
# trt.TacticSource.CUBLAS_LT                -> 1, deprecated
# trt.TacticSource.CUDNN                    -> 2, deprecated
# trt.TacticSource.EDGE_MASK_CONVOLUTIONS   -> 3
# trt.TacticSource.JIT_CONVOLUTIONS         -> 4

print(f"{builder_config.profiling_verbosity = }")  # Get/set profiling verbosity
# Alternative values of trt.ProfilingVerbosity:
# trt.ProfilingVerbosity.LAYER_NAMES_ONLY   -> 0, default
# trt.ProfilingVerbosity.NONE               -> 1
# trt.ProfilingVerbosity.DETAILED           -> 2

print(f"{builder_config.remote_auto_tuning_config = }")  # Get/set remote auto-tuning configuration, default: ""
# This is only used for `builder_config.engine_capability = trt.EngineCapability.SAFETY`
# A example of remote auto-tuning configuration:
# "ssh://wili:wili@10.19.23.29:22?remote_exec_path=/usr/local/bin&remote_lib_path=/usr/lib/x86_64-linux-gnu&dump_remote_stdout=on&dump_remote_stderr=on"

timing_cache = tw.builder_config.create_timing_cache(b"")
tw.builder_config.set_timing_cache(timing_cache, False)  # Set timing cache, 04-Feature/TimingCache
tw.builder_config.get_timing_cache()  # Get timing cache, 04-Feature/TimingCache

print("Finish")
