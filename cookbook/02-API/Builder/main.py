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
builder.reset()  # reset Builder as default
#builder.max_batch_size = 6 # use in Implicit Batch Mode, deprecated since TensorRT 8.4, use Dynamic Shape Mode instead
#builder.max_workspace_size = 1 << 30  # deprecated since TensorRT 8.4, use BuilderConfig.set_memory_pool_limit instead
max_threads = 16  # the most threads to use the Builder
print("builder.__sizeof__() = %d" % builder.__sizeof__())
print("builder.__str__() = %s" % builder.__str__())
print("builder.logger = %s" % builder.logger)
print("builder.platform_has_tf32 = %s" % builder.platform_has_tf32)
print("builder.platform_has_fast_fp16 = %s" % builder.platform_has_fast_fp16)
print("builder.platform_has_fast_int8 = %s" % builder.platform_has_fast_int8)
print("builder.num_DLA_cores = %d" % builder.num_DLA_cores)
print("builder.max_DLA_batch_size = %d" % builder.max_DLA_batch_size)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))

print("builder.is_network_supported() = %s" % builder.is_network_supported(network, config))

engineString = builder.build_serialized_network(network, config)
#engine = builder.build_engine(network, config)  # deprecate since TensorRT 8.0, use build_serialized_network instead
'''
ICudaEngine 的成员方法
++++ 表示代码中进行了用法展示
==== 表示代码中作为 binding 部分进行了用法展示
---- 表示代码中没有进行用法展示
无前缀表示其他内部方法

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
++++build_engine
++++build_serialized_network
++++create_builder_config
++++create_network
++++create_optimization_profile
----error_recorder 见 09-Advanve/ErrorRecorder
----gpu_allocator 见 09-Advanve/GPUAllocator
++++is_network_supported
++++logger
++++max_DLA_batch_size
++++max_batch_size
++++max_threads
++++num_DLA_cores
++++platform_has_fast_fp16
++++platform_has_fast_int8
++++platform_has_tf32
++++reset
'''
