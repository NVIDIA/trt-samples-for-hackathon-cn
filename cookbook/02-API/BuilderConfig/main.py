#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from cuda import cudart
import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.reset()  # 还原 BuidlerConfig 默认配置，非必需 API
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
print("config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM))
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM))
print("config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM) = %d" % config.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM))

print("config.__sizeof__() = %d" % config.__sizeof__())
print("config.__str__() = %s" % config.__str__())
print("config.default_device_type = %s" % config.default_device_type)
print("config.can_run_on_DLA = %s" % config.can_run_on_DLA)
print("config.DLA_core = %d" % config.DLA_core)

config.engine_capability = trt.EngineCapability.STANDARD  # 默认值，无约束，允许使用 GPU 或 DLA
#config.engine_capability = trt.EngineCapability.SAFETY # 只允许使用 GPU 上的 Safrty Runtime
#config.engine_capability = trt.EngineCapability.DLA_STANDALONE # 只允许使用 DLA
#config.engine_capability = trt.EngineCapability.DEFAULT # 同 STANDARD，deprecated since TensorRT 8.0
#config.engine_capability = trt.EngineCapability.SAFE_GPU # 同 SAFETY，deprecated since TensorRT 8.0
#config.engine_capability = trt.EngineCapability.SAFE_DLA # 同 DLA_STANDALONE，deprecated since TensorRT 8.0
print("config.engine_capability = %s" % config.engine_capability)

print("config.flags = %d" % config.flags)
print("Set Flag FP16!")
config.set_flag(trt.BuilderFlag.FP16)
print("config.get_flag(trt.BuilderFlag.FP16) = %s" % config.get_flag(trt.BuilderFlag.FP16))
print("config.flags = %d" % config.flags)
print("Clear Flag FP16!")
config.clear_flag(trt.BuilderFlag.FP16)
print("config.flags = %d" % config.flags)
print("Set Flag by bit operation!")
config.flags = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INT8)
print("config.flags = %d" % config.flags)
config.flags = 0
print("config.flags = %d" % config.flags)
print("trt.BuilderFlag.FP16 = %d" % int(trt.BuilderFlag.FP16))
print("trt.BuilderFlag.INT8 = %d" % int(trt.BuilderFlag.INT8))
print("trt.BuilderFlag.DEBUG = %d" % int(trt.BuilderFlag.DEBUG))
print("trt.BuilderFlag.GPU_FALLBACK = %d" % int(trt.BuilderFlag.GPU_FALLBACK))
print("trt.BuilderFlag.STRICT_TYPES = %d" % int(trt.BuilderFlag.STRICT_TYPES))
print("trt.BuilderFlag.REFIT = %d" % int(trt.BuilderFlag.REFIT))
print("trt.BuilderFlag.DISABLE_TIMING_CACHE = %d" % int(trt.BuilderFlag.DISABLE_TIMING_CACHE))
print("trt.BuilderFlag.TF32 = %d" % int(trt.BuilderFlag.TF32))
print("trt.BuilderFlag.SPARSE_WEIGHTS = %d" % int(trt.BuilderFlag.SPARSE_WEIGHTS))
print("trt.BuilderFlag.SAFETY_SCOPE = %d" % int(trt.BuilderFlag.SAFETY_SCOPE))
print("trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS = %d" % int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS))
print("trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS = %d" % int(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS))
print("trt.BuilderFlag.DIRECT_IO = %d" % int(trt.BuilderFlag.DIRECT_IO))
print("trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS = %d" % int(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS))

print("config.quantization_flags = %d" % config.quantization_flags)
print("Set Flag CALIBRATE_BEFORE_FUSION!")
config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
print("config.get_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION) = %s" % config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION))
print("config.quantization_flags = %d" % config.quantization_flags)
print("Clear Flag CALIBRATE_BEFORE_FUSION!")
config.clear_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
print("config.quantization_flags = %d" % config.quantization_flags)

inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

print("config.num_optimization_profiles = %d" % config.num_optimization_profiles)
config.profile_stream = 0  # 指定用于自动优化的 CUDA Stream，默认值 0 使用 CUDA Default Stream
config.avg_timing_iterations = 10  # 自动优化阶段每个 Layer 的每个 Tactic 的平均测试运行次数，默认值 1
#config.min_timing_iterations = 2  # 自动优化阶段每个 Layer 的每个 Tactic 的最少测试运行次数，默认值 1，deprecated since TensorRT 8.4
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

config.set_device_type(identityLayer, trt.DeviceType.DLA)
print("config.get_device_type(identityLayer) = %s" % config.get_device_type(identityLayer))  # 网络中某层运行的设备
print("config.is_device_type_set(identityLayer) = %s" % config.is_device_type_set(identityLayer))
config.reset_device_type(identityLayer)
print("config.get_device_type(identityLayer) = %s" % config.get_device_type(identityLayer))

engineString = builder.build_serialized_network(network, config)
'''
ICudaEngine 的成员方法
++++ 表示代码中进行了用法展示
==== 表示代码中作为 binding 部分进行了用法展示
---- 表示代码中没有进行用法展示
无前缀表示其他内部方法

DLA_core
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
__reduce__
__reduce_ex__
__repr__
__setattr__
++++__sizeof__
++++__str__
__subclasshook__
++++add_optimization_profile
----algorithm_selector 见 09-Advance/AlgorithmSelector
++++avg_timing_iterations
++++can_run_on_DLA
++++clear_flag
++++clear_quantization_flag
----create_timing_cache 见 09-Advance/TimingCache
++++default_device_type
++++engine_capability
++++flags
----get_calibration_profile 见 02-API/Int8-PTQ
++++get_device_type
++++get_flag
++++get_memory_pool_limit
++++get_quantization_flag
----get_tactic_sources 见 09-Advance/TacticSource
----get_timing_cache 见 09-Advance/TimingCache
----int8_calibrator int8 模式需要，见 03-APIModel/MNISTExample-pyTorch/main.py
----is_device_type_set
----max_workspace_size deprecated since TensorRT 8.0, use get_memory_pool_limit instead
++++min_timing_iterations
++++num_optimization_profiles
++++profile_stream
----profiling_verbosity 见 09-Advance/ProfilingVerbosity
++++quantization_flags
++++reset
++++reset_device_type
----set_calibration_profile int8 模式需要，见 03-APIModel/MNISTExample-pyTorch/main.py
++++set_device_type
++++set_flag
++++set_memory_pool_limit
++++set_quantization_flag
----set_tactic_sources 见 09-Advance/TacticSource
----set_timing_cache 见 09-Advance/TimingCache

~~~~~~~~ API since TensorRT 8.5 ~~~~~~~~
get_preview_feature
set_preview_feature
'''
