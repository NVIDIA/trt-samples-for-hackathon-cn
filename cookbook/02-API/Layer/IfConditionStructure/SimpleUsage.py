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

nB0, nC0, nH0, nW0 = 1, 3, 4, 5  # 输入张量 NCHW
data0 = np.arange(1, 1 + nB0 * nC0 * nH0 * nW0, dtype=np.float32).reshape(nB0, nC0, nH0, nW0)  # 输入数据
data1 = data0 - 1

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
inputT0 = network.add_input("inputT0", trt.float32, (nB0, nC0, nH0, nW0))
#-------------------------------------------------------------------------------# 网络部分
# 以 “inputT0.reshape(-1)[0] != 0” 作为判断条件
_H0 = network.add_slice(inputT0, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
_H1 = network.add_reduce(_H0.get_output(0), trt.ReduceOperation.SUM, (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3), False)
_H2 = network.add_identity(_H1.get_output(0))
_H2.set_output_type(0, trt.bool)
_H2.get_output(0).dtype = trt.bool

# 添加 condition 层
ifCondition = network.add_if_conditional()
ifConditionInputLayer = ifCondition.add_input(inputT0)
ifConditionConditionLayer = ifCondition.set_condition(_H2.get_output(0))  # 条件必须是 0 维 bool 型张量

# 判断条件成立时的分支
_H3 = network.add_elementwise(ifConditionInputLayer.get_output(0), ifConditionInputLayer.get_output(0), trt.ElementWiseOperation.SUM)

# 判断条件不成立时的分支
_H4 = network.add_identity(ifConditionInputLayer.get_output(0))

ifConditionOutputLayer = ifCondition.add_output(_H3.get_output(0), _H4.get_output(0))
#-------------------------------------------------------------------------------# 网络部分
network.mark_output(ifConditionOutputLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()

nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data0)  # 满足 if 条件的输入，计算结果是输入张量的 2 倍
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

bufferH[0] = data1  # 不满足 if 条件的输入，计算结果是保持输入张量不变

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