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

import numpy as np
from cuda import cudart
import tensorrt as trt

nB, nC, nH, nW = 1, 4, 8, 8
data = (np.arange(nB * nC * nH * nW, dtype=np.float32) / np.prod(nB * nC * nH * nW) * 128).astype(np.float32).reshape(nB, nC, nH, nW)
np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
profile.set_shape(inputT0.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)
layer = network.add_identity(inputT0)
layer.precision = trt.int8
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.get_output(0).dynamic_range = [-128, 128]
network.mark_output(layer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # count of input / output tensor
nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

context = engine.create_execution_context()
context.debug_sync = True  # debug switch, one VERBOSE log will be added each time context.execute_v2() was called.
context.nvtx_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY  # defaut value
#context.nvtx_verbosity = trt.ProfilingVerbosity.kNONE
#context.nvtx_verbosity = trt.ProfilingVerbosity.kDETAILED
#context.nvtx_verbosity = trt.ProfilingVerbosity.kDEFAULT  # same as LAYER_NAMES_ONLY, deprecated Since TensorRT 8.5
#context.nvtx_verbosity = trt.ProfilingVerbosity.kVERBOSE  # same as kDETAILED, deprecated Since TensorRT 8.5

print("context.__sizeof__() = %d" % context.__sizeof__())
print("context.__str__() = %s" % context.__str__())
print("context.name = %s" % context.name)
print("context.engine = %s" % context.engine)
print("context.enqueue_emits_profile = %s" % context.enqueue_emits_profile)  # refer to 09-Advance/Profiler
print("context.active_optimization_profile = %d" % context.active_optimization_profile)
print("context.persistent_cache_limit = %d" % context.persistent_cache_limit)

print("context.infer_shapes() = %s" % context.infer_shapes())
print("context.all_binding_shapes_specified = %s" % context.all_binding_shapes_specified)  # only work for set_binding_shape(), bot for set_input_shape()
print("context.all_shape_inputs_specified = %s" % context.all_shape_inputs_specified)  # only work for set_input_shape(), not for set_shape_input()
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), context.get_tensor_shape(lTensorName[i]))
    #print("%s[%2d]->" % ("Input " if i < nInput else "Output", i), context.get_binding_shape(lTensorName[i]))
context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
#context.set_binding_shape(0, [nB, nC, nH, nW])
print("context.infer_shapes() = %s" % context.infer_shapes())
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), context.get_tensor_shape(lTensorName[i]), context.get_tensor_strides(lTensorName[i]))
    #print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), context.get_binding_shape(i), context.get_strides(i))
    #print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), context.get_shape(i))  # no input shape tensor in this code example

for i in range(nInput, nIO):
    print("[%2d]Output->" % i, context.get_max_output_size(lTensorName[i]))

bufferH = []
bufferH.append(data)
for i in range(nOutput):
    bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    print(context.get_tensor_address(lTensorName[i]))

#context.execute(nB, bufferD)  # deprecated since TensorRT 7.0, use for Implicit Batch mode
context.execute_v2(bufferD)
context.execute_async_v2(bufferD, 0)
context.execute_async_v3(0)

for i in range(nOutput):
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput):
    print("Input %d:" % i, bufferH[i].shape, "\n", bufferH[i])
for i in range(nOutput):
    print("Output %d:" % i, bufferH[nInput + i].shape, "\n", bufferH[nInput + i])

print("Restore to Linear:")
print(bufferH[-1].reshape(nB * nC * nH * 2, nW // 2).transpose(1, 0).reshape(nB, nC, nH, nW))

for buffer in bufferD:
    cudart.cudaFree(buffer)
"""
Member of IExecutionContext:
++++        shown above
====        shown in binding part
~~~~        deprecated
----        not shown above
[no prefix] others

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
++++active_optimization_profile
++++all_binding_shapes_specified
++++all_shape_inputs_specified
++++debug_sync
----device_memory unreadable attribution
++++engine
++++enqueue_emits_profile
----error_recorder refer to 09-Advance/ErrorRecorder
++++execute
++++execute_async
++++execute_async_v2
++++execute_async_v3
++++execute_v2
++++get_binding_shape
----get_input_consumed_event refer to 09-Advance/Event
++++get_max_output_size
----get_output_allocator refer to 09-Advance/OutputAllocator
----get_shape no input shape tensor in this code example, refer to 02-API/Layer/ShufleLayer/DynamicShuffleWithShapeTensor.py
++++get_strides
++++get_tensor_address
++++get_tensor_shape
++++get_tensor_strides
++++infer_shapes
++++name
++++nvtx_verbosity
++++persistent_cache_limit
----profiler refer to 9-Advance/Profiler
----report_to_profiler refer to 9-Advance/Profiler
++++set_binding_shape
----set_input_consumed_event refer to 09-Advance/Event
++++set_input_shape
----set_optimization_profile_async refer to 09-Advance/MultiOptimizationProfile
----set_output_allocator refer to 09-Advance/OutputAllocator
----set_shape_input refer to 02-API/Layer/ShufleLayer/DynamicShuffleWithShapeTensor.py
++++set_tensor_address
----temporary_allocator refer to 09-Advance/GPUAllocator
"""