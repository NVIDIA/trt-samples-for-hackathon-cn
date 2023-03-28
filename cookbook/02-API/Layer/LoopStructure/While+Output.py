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
data = np.ones([nB, nC, nH, nW], dtype=np.float32)
length = 7

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
#-------------------------------------------------------------------------------
loop = network.add_loop()  # 添加 Loop 结构
rLayer = loop.add_recurrence(inputT0)  # 循环入口

_H1 = network.add_reduce(rLayer.get_output(0), trt.ReduceOperation.MAX, (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3), False)  # 取循环体张量的第一个元素，判断其是否小于 6
_H2 = network.add_constant((), np.array([6], dtype=np.float32))
_H3 = network.add_elementwise(_H2.get_output(0), _H1.get_output(0), trt.ElementWiseOperation.SUB)
_H4 = network.add_activation(_H3.get_output(0), trt.ActivationType.RELU)
_H5 = network.add_identity(_H4.get_output(0))
_H5.set_output_type(0, trt.bool)
_H5.get_output(0).dtype = trt.bool
loop.add_trip_limit(_H5.get_output(0), trt.TripLimit.WHILE)  # 判断结果转为 BOOL 类型，交给 TripLimit

_H0 = network.add_scale(rLayer.get_output(0), trt.ScaleMode.UNIFORM, np.array([1], dtype=np.float32), np.array([1], dtype=np.float32), np.array([1], dtype=np.float32))  # 循环体，给输入元素加 1
rLayer.set_input(1, _H0.get_output(0))

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # 第一种循环输出，只保留最终结果，index 参数被忽略
loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.CONCATENATE, 0)  # 第二种循环输出，保留所有中间结果，传入 rLayer 则保留“第 0 到第 t-1 次迭代的结果”（类比 while 循环），传入 _H0 则保留“第 1 到第 t 次迭代的结果”（类比 do-while 循环，不推荐使用，可能有错误）
lengthLayer = network.add_constant((), np.array([length], dtype=np.int32))
loopOutput1.set_input(1, lengthLayer.get_output(0))  # 指定需要保留的长度，若这里传入张量的值 v <= t，则结果保留前 v 次迭代，若 v > t，则多出部分用 0 填充
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