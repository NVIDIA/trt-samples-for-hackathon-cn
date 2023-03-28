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

import tensorrt as trt

def printFlagFromBit(bit):
    flagList = []
    if bit & 1 << int(trt.BuilderFlag.FP16):  # 0
        flagList.append("FP16")
    if bit & 1 << int(trt.BuilderFlag.INT8):  # 1
        flagList.append("INT8")
    if bit & 1 << int(trt.BuilderFlag.DEBUG):  # 2
        flagList.append("DEBUG")
    if bit & 1 << int(trt.BuilderFlag.GPU_FALLBACK):  # 3
        flagList.append("GPU_FALLBACK")
    if bit & 1 << int(trt.BuilderFlag.STRICT_TYPES):  # 4
        flagList.append("STRICT_TYPES")
    if bit & 1 << int(trt.BuilderFlag.REFIT):  # 5
        flagList.append("REFIT")
    if bit & 1 << int(trt.BuilderFlag.DISABLE_TIMING_CACHE):  # 6
        flagList.append("DISABLE_TIMING_CACHE")
    if bit & 1 << int(trt.BuilderFlag.TF32):  # 7
        flagList.append("TF32")
    if bit & 1 << int(trt.BuilderFlag.SPARSE_WEIGHTS):  # 8
        flagList.append("SPARSE_WEIGHTS")
    if bit & 1 << int(trt.BuilderFlag.SAFETY_SCOPE):  # 9
        flagList.append("SAFETY_SCOPE")
    if bit & 1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS):  # 10
        flagList.append("OBEY_PRECISION_CONSTRAINTS")
    if bit & 1 << int(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS):  # 11
        flagList.append("PREFER_PRECISION_CONSTRAINTS")
    if bit & 1 << int(trt.BuilderFlag.DIRECT_IO):  # 12
        flagList.append("DIRECT_IO")
    if bit & 1 << int(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS):  # 13
        flagList.append("REJECT_EMPTY_ALGORITHMS")
    if bit & 1 << int(trt.BuilderFlag.ENABLE_TACTIC_HEURISTIC):  # 14
        flagList.append("ENABLE_TACTIC_HEURISTIC")
    if bit & 1 << int(trt.BuilderFlag.VERSION_COMPATIBLE):  # 15
        flagList.append("VERSION_COMPATIBLE")
    if bit & 1 << int(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME):  # 16
        flagList.append("EXCLUDE_LEAN_RUNTIME")
    if bit & 1 << int(trt.BuilderFlag.FP8):  # 17
        flagList.append("FP8")

    print(flagList)
    return

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.reset()  # reset BuidlerConfig as default, not required

inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

print("config.__sizeof__() = %d" % config.__sizeof__())
print("config.__str__() = %s" % config.__str__())

print("\nDevice type part ======================================================")
config.engine_capability = trt.EngineCapability.STANDARD  # default without targeting safety runtime, supporting GPU and DLA
#config.engine_capability = trt.EngineCapability.SAFETY # targeting safety runtime, supporting GPU on NVIDIA Drive(R) products
#config.engine_capability = trt.EngineCapability.DLA_STANDALONE # targeting DLA runtime, supporting DLA
#config.engine_capability = trt.EngineCapability.DEFAULT # same as STANDARD, deprecated since TensorRT 8.0
#config.engine_capability = trt.EngineCapability.SAFE_GPU # same as SAFETY, deprecated since TensorRT 8.0
#config.engine_capability = trt.EngineCapability.SAFE_DLA # same as DLA_STANDALONE, deprecated since TensorRT 8.0
print("config.engine_capability = %s" % config.engine_capability)

print("config.default_device_type = %s" % config.default_device_type)
print("config.DLA_core = %d" % config.DLA_core)
print("config.can_run_on_DLA(identityLayer) = %s" % config.can_run_on_DLA(identityLayer))
print("Set device type of certain layer ----------------------------------------")
config.set_device_type(identityLayer, trt.DeviceType.DLA)  # device type: [trt.DeviceType.GPU, trt.DeviceType.DLA]
print("config.get_device_type(identityLayer) = %s" % config.get_device_type(identityLayer))  # offload one layer running on certain device
print("config.is_device_type_set(identityLayer) = %s" % config.is_device_type_set(identityLayer))
print("Reset device type of certain layer to default ---------------------------")
config.reset_device_type(identityLayer)
print("config.get_device_type(identityLayer) = %s" % config.get_device_type(identityLayer))

print("\nFlag part =============================================================")
print("config.flags = %d" % config.flags)  # check all flags, when running TensorRT on Ampere above, TF32 (1<<7) is set as default
printFlagFromBit(config.flags)

print("Set Flag FP16 -----------------------------------------------------------")
config.set_flag(trt.BuilderFlag.FP16)  # set single flag
print("config.get_flag(trt.BuilderFlag.FP16) = %s" % config.get_flag(trt.BuilderFlag.FP16))  # check single flag
printFlagFromBit(config.flags)

print("Clear Flag FP16 ---------------------------------------------------------")
config.clear_flag(trt.BuilderFlag.FP16)  # unset single flag
print("config.get_flag(trt.BuilderFlag.FP16) = %s" % config.get_flag(trt.BuilderFlag.FP16))  # check single flag
printFlagFromBit(config.flags)

print("Set Flag by bit operation -----------------------------------------------")
config.flags = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INT8)  # set multiple flags
printFlagFromBit(config.flags)
config.flags = 0  # unset all flags
printFlagFromBit(config.flags)

print("config.quantization_flags = %d" % config.quantization_flags)  # check quantization flag
print("Set flag CALIBRATE_BEFORE_FUSION ----------------------------------------")
config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # set quantization flag
print("config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) = %s" % config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION))
print("config.quantization_flags = %d" % config.quantization_flags)

print("Clear flag CALIBRATE_BEFORE_FUSION --------------------------------------")
config.clear_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)  # unset quantization flag
print("config.quantization_flags = %d" % config.quantization_flags)

print("\nPreview feature part ==================================================")
config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
print("config.get_preview_feature(FASTER_DYNAMIC_SHAPES_0805) = %d" % config.get_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805))
# available vavaluesle:
#config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, True)
#config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

print("\nEngine build part =====================================================")
print("config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = %d Byte (%.1f GiB)" % (config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE), config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) / (1 << 30)))  # all GPU memory is occupied by default
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM))
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM))
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM))
print("config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) = %d Byte (%.1f GiB)" % (config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM), config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) / (1 << 30)))

print("Set workspace manually---------------------------------------------------")
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
print("config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = %d Byte (%.1f GiB)" % (config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE), config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) / (1 << 30)))
print("config.num_optimization_profiles = %d" % config.num_optimization_profiles)
print("config.builder_optimization_level = %d" % config.builder_optimization_level)  # optimzation level of autotuning from 0 (shortest building time) to 5 (best performance)
config.profile_stream = 0  # set the CUDA stream for auto tuning, default value is 0
config.avg_timing_iterations = 10  # average times to running each tactic for auto tuning, default value is 1
#config.min_timing_iterations = 1  # minimum times to running each tactic for auto tuning, default value is 1, deprecated since TensorRT 8.4

print("config.hardware_compatibility_level = %d" % config.hardware_compatibility_level)
# available values:
#config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
#config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
print("config.max_aux_streams = %d" % config.max_aux_streams)
print("config.plugins_to_serialize =", config.plugins_to_serialize)

engineString = builder.build_serialized_network(network, config)
"""
Member of IBuilderConfig:
++++        shown above
----        not shown above
[no prefix] others

++++DLA_core
----__class__
__del__
__delattr__
__dir__
__doc__
__enter__
__eq__
__exit__
__format__
__ge__
__getattribute__
__gt__
__hash__
__init__
__init_subclass__
__le__
__lt__
__module__
__ne__
__new__
----__pybind11_module_local_v4_gcc_libstdcpp_cxxabi1013__
__reduce__
__reduce_ex__
__repr__
__setattr__
++++__sizeof__
++++__str__
__subclasshook__
++++add_optimization_profile
----algorithm_selector                                                          refer to 09-Advance/AlgorithmSelector
++++avg_timing_iterations
++++builder_optimization_level
++++can_run_on_DLA
++++clear_flag
++++clear_quantization_flag
----create_timing_cache                                                         refer to 09-Advance/TimingCache
++++default_device_type
++++engine_capability
++++flags
----get_calibration_profile                                                     refer to 02-API/Int8-PTQ
++++get_device_type
++++get_flag
++++get_memory_pool_limit
++++get_preview_feature
++++get_quantization_flag
----get_tactic_sources                                                          refer to 09-Advance/TacticSource
----get_timing_cache                                                            refer to 09-Advance/TimingCache
++++hardware_compatibility_level
----int8_calibrator needed by INT8 mode,                                        refer to 03-APIModel/MNISTExample-pyTorch/main.py
----is_device_type_set
++++max_aux_streams                                                             refer to 09-Advance/AuxStream
----max_workspace_size deprecated since TensorRT 8.0, use get_memory_pool_limit instead
++++min_timing_iterations
++++num_optimization_profiles
++++plugins_to_serialize                                                        refer to 05-Plugin/PluginSerialize
++++profile_stream
----profiling_verbosity                                                         refer to 09-Advance/ProfilingVerbosity
++++quantization_flags
++++reset
++++reset_device_type
----set_calibration_profile                                                     needed by INT8 mode, refer to 03-APIModel/MNISTExample-pyTorch/main.py
++++set_device_type
++++set_flag
++++set_memory_pool_limit
++++set_preview_feature
++++set_quantization_flag
----set_tactic_sources                                                          refer to 09-Advance/TacticSource
----set_timing_cache                                                            refer to 09-Advance/TimingCache
"""