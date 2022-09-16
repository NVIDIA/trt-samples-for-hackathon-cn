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

nB, nC, nH, nW = 1, 4, 8, 8  # nC % 4 ==0，全部值得到保存
#nB, nC, nH, nW = 1, 3, 8, 8  # nC % 4 !=0，会丢值
data = (np.arange(1, 1 + nB * nC * nH * nW, dtype=np.float32) / np.prod(nB * nC * nH * nW) * 128).astype(np.float32).reshape(nB, nC, nH, nW)

np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
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
network.unmark_output(layer.get_output(0))
network.mark_output(layer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
context.set_binding_shape(0, [nB, nC, nH, nW])

context.debug_sync = True  # 打开 debug 开关（默认关闭），每次 context.execute_v2 执行后将会在日志中添加一条 VERBOSE 记录

print("context.__sizeof__() = %d" % context.__sizeof__())
print("context.__str__() = %s" % context.__str__())
print("context.engine = %s" % context.engine)
print("context.enqueue_emits_profile = %s" % context.enqueue_emits_profile)
print("context.active_optimization_profile = %d" % context.active_optimization_profile)
print("context.all_binding_shapes_specified = %s" % context.all_binding_shapes_specified)
#print("context.all_shape_inputs_specified = %d" % context.all_shape_inputs_specified)  # 范例中没有用到 Shape Input Tensor

bufferH = []
bufferH.append(data)
for i in range(nOutput):
    bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

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
IExecutionContext 的成员方法
++++ 表示代码中进行了用法展示
---- 表示代码中没有进行展示
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
++++active_optimization_profile 见 09-Advance/MultiOptimizationProfile
++++all_binding_shapes_specified
----all_shape_inputs_specified 针对 Shape Tensor 的 all_binding_shapes_specified，见 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
++++debug_sync
----device_memory 不可读属性
++++engine
----enqueue_emits_profile 要配合 Profiler 使用，见 09-Advance/Profiler
----error_recorder 见 09-Advanve/ErrorRecorder
----execute 用于 Implicite Batch 模式，已经废弃，见 01-SimpleDemo/TensorRT7
----execute_async 用于 Implicite Batch 模式的异步执行，已经废弃
++++execute_v2
----execute_async_v2 异步版本的 execute_v2，见 09-Advance/StreamAndAsync

~~~~~~~~ API since TensorRT8.5 ~~~~~~~~
execute_async_v3
get_input_consumed_event
get_max_output_size
get_output_allocator
get_tensor_address
get_tensor_shape
get_tensor_strides
infer_shapes
nvtx_verbosity
persistent_cache_limit
set_input_consumed_event
set_input_shape
set_output_allocator
set_tensor_address
temporary_allocator
"""
