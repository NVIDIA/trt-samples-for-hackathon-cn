#
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

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(0)
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.reset()  # reset BuidlerConfig as default

inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

layer = network.add_identity(inputTensor)
network.mark_output(layer.get_output(0))

print(f"\n================================================================ Device related")
print(f"{config.engine_capability = }")
# Alternative values of trt.EngineCapability:
# trt.EngineCapability.STANDARD         -> 0, default without targeting safety runtime, supporting GPU and DLA
# trt.EngineCapability.SAFETY           -> 1, targeting safety runtime, supporting GPU on NVIDIA Drive(R) products
# trt.EngineCapability.DLA_STANDALONE   -> 2, targeting DLA runtime, supporting DLA
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

print(f"\n================================================================ Flag related")

def print_flag():
    flags = []
    for name, (value, text) in trt.BuilderFlag.__entries.items():
        flags.append([value, name])
    flags = sorted(flags, key=lambda x: x[0])
    for value, name in flags:
        print(f"{int(value):3d}: {name}")
    return

print_flag()  # print all flags

print(f"{config.flags = }")  # get or set flags, TF32 (1<<6) is set as default on Ampere above GPU
config.set_flag(trt.BuilderFlag.FP16)  # set single flag
config.clear_flag(trt.BuilderFlag.FP16)  # unset single flag
config.flags = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INT8)  # set multiple flags
config.flags = 0  # unset all flags

print(f"{config.quantization_flags = }")  # get or set quantization flags, 0 as default
print(f"{config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) = }")  # get whether the quantization flag is enabled
config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # set single quantization flag
config.clear_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # clear single quantization flag

print(f"\n================================================================ Preview feature related")
print(f"{config.get_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806)}")  # check whether the preview feature is enabled, this switch is enabled as default in TensorRT-10 and might be removed in the future
config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

print(f"\n================================================================ Engine related")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = } Bytes")  # all GPU memory is used by default
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM) = } Bytes")
print(f"{config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM) = } Bytes")
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

print(f"{config.num_optimization_profiles = }")  # get number of Optimization-Profile, default value: 0
print(f"{config.builder_optimization_level = }")  # get or set optimization level, default value: 3
print(f"{config.profile_stream = }")  # get or set the CUDA stream for auto tuning, default value: 0
print(f"{config.avg_timing_iterations = }")  # get or set average times to running each tactic during auto tuning, default value: 1
print(f"{config.hardware_compatibility_level = }")  # get or set hardware compatibility level, default value: trt.HardwareCompatibilityLevel.NONE
# Alternative values of trt.HardwareCompatibilityLevel:
# trt.HardwareCompatibilityLevel.NONE           -> 0
# trt.HardwareCompatibilityLevel.AMPERE_PLUS    -> 1

print(f"{config.int8_calibrator = }")  # get or set INT8 calibreator
print(f"{config.get_calibration_profile() = }")  # get INT8 calibration profile
config.set_calibration_profile(profile)  # set INT8 calibration profile

print(f"{config.algorithm_selector = }")  # get or set algorithm selector
print(f"{config.progress_monitor = }")  # get or set algorithm selector
print(f"{config.max_aux_streams = }")  # get or set auxiliary CUDA streams to do inference, default value: -1
print(f"{config.plugins_to_serialize = }")

print(f"{config.get_tactic_sources() = }")  # get tactic sources, default value: 24
config.set_tactic_sources(0)
# Alternative values of trt.TacticSource:
# trt.TacticSource.CUBLAS                   -> 0, deprecated
# trt.TacticSource.CUBLAS_LT                -> 1, deprecated
# trt.TacticSource.CUDNN                    -> 2, deprecated
# trt.TacticSource.EDGE_MASK_CONVOLUTIONS   -> 3
# trt.TacticSource.JIT_CONVOLUTIONS         -> 4
# Input argument of set_tactic_sources() is a bit mask of trt.TacticSource

print(f"{config.profiling_verbosity = }")  # get or set profiling verboseity
# Alternative values of trt.ProfilingVerbosity:
# trt.ProfilingVerbosity.LAYER_NAMES_ONLY   -> 0, default
# trt.ProfilingVerbosity.NONE               -> 1
# trt.ProfilingVerbosity.DETAILED           -> 2

builder.build_serialized_network(network, config)

print("Finish")
"""
APIs not showed:
create_timing_cache -> 04-Feature/TimingCache
get_timing_cache    -> 04-Feature/TimingCache
set_timing_cache    -> 04-Feature/TimingCache
"""
