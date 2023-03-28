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

import numpy as np
from cuda import cudart
import tensorrt as trt

nB, nC, nH, nW = 1, 3, 4, 5
data = np.ones([nB, nC, nH, nW], dtype=np.float32) * np.arange(1, 1 + nC, dtype=np.float32).reshape(1, nC, 1, 1)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
#-------------------------------------------------------------------------------
loop = network.add_loop()
iteratorLayer = loop.add_iterator(inputT0, 1, False)  # 制造一个迭代器，在 C 维上每次正向抛出 1 层 (1,nH,nW)
iteratorLayer.axis = 1  # 重设抛出的轴号，最高维为 0，往低维递增
print(iteratorLayer.reverse)  # 是否反序抛出（见后面范例），仅用于输出不能修改，这里会在运行时输出 False

limit = network.add_constant((), np.array([nC], dtype=np.int32))
loop.add_trip_limit(limit.get_output(0), trt.TripLimit.COUNT)

_H0 = network.add_constant([1, nH, nW], np.ones(nH * nW, dtype=np.float32))  # 首次循环前的循环体输入张量，必须在循环外初始化好，这里相当于求和的初始值
rLayer = loop.add_recurrence(_H0.get_output(0))

_H1 = network.add_elementwise(rLayer.get_output(0), iteratorLayer.get_output(0), trt.ElementWiseOperation.SUM)
rLayer.set_input(1, _H1.get_output(0))

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # 只保留最后输出，index 参数被忽略
loopOutput1 = loop.add_loop_output(_H1.get_output(0), trt.LoopOutput.CONCATENATE, 0)  # 保留所有中间输出，index 可以使用其他参数（例子见后面）
lengthLayer = network.add_constant((), np.array([nC], dtype=np.int32))
loopOutput1.set_input(1, lengthLayer.get_output(0))
#-------------------------------------------------------------------------------
network.mark_output(loopOutput0.get_output(0))
network.mark_output(loopOutput1.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

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

for buffer in bufferD:
    cudart.cudaFree(buffer)