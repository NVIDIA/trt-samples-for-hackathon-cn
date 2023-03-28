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

import os

import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = "./model.plan"
data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, "rb") as f:
        engineString = f.read()
    if engineString == None:
        print("Failed getting serialized engine!")
        exit()
    print("Succeeded getting serialized engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
    profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
    config.add_optimization_profile(profile)

    identityLayer = network.add_identity(inputTensor)
    network.mark_output(identityLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building serialized engine!")
        exit()
    print("Succeeded building serialized engine!")
    with open(trtFile, "wb") as f:
        f.write(engineString)
        print("Succeeded saving .plan file!")

runtime = trt.Runtime(logger)

print("runtime.__sizeof__() = %d" % runtime.__sizeof__())
print("runtime.__str__() = %s" % runtime.__str__())

print("\nRuntime related =======================================================")
print("runtime.logger = %s" % runtime.logger)
print("runtime.DLA_core = %d" % runtime.DLA_core)
print("runtime.num_DLA_cores = %d" % runtime.num_DLA_cores)
print("runtime.engine_host_code_allowed = %s" % runtime.engine_host_code_allowed)

runtime.max_threads = 16  # The maximum thread that can be used by the Runtime
tempfile_control_flags = trt.TempfileControlFlag.ALLOW_IN_MEMORY_FILES
# available values
#tempfile_control_flags = trt.TempfileControlFlag.ALLOW_TEMPORARY_FILES

temporary_directory = "."

engine = runtime.deserialize_cuda_engine(engineString)
"""
Member of IExecutionContext:
++++        shown above
====        shown in binding part
~~~~        deprecated
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
++++deserialize_cuda_engine
++++engine_host_code_allowed
error_recorder                                                                  refer to 09-Advance/ErrorRecoder
get_plugin_registry
gpu_allocator                                                                   refer to 09-Advance/GPUAllocator
load_runtime
++++logger
++++max_threads
++++num_DLA_cores
++++tempfile_control_flags
++++temporary_directory
"""