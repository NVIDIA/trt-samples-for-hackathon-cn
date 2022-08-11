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

nB, nC, nH, nW = 1, 3, 4, 5  # 输入张量 NCHW
data0 = np.arange(nB * nC * nH * nW, dtype=np.float32).reshape(nB, nC, nH, nW)  # 输入数据
data1 = np.arange(nB * nC, dtype=np.float32).reshape(nB, nC)  # 输入数据
data2 = np.arange(nB * (nC + 1), dtype=np.float32).reshape(nB, nC + 1)  # 输入数据

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, -1, 5))
profile.set_shape(inputT0.name, (1, 1, 1, 5), (1, 3, 4, 5), (2, 6, 8, 5))
inputT1 = network.add_input("inputT1", trt.float32, (-1, -1))
profile.set_shape(inputT1.name, (1, 1), (1, 3), (2, 6))
config.add_optimization_profile(profile)
#-------------------------------------------------------------------------------# 网络部分
_H1 = network.add_shape(inputT0)
_H2 = network.add_slice(_H1.get_output(0), [1], [1], [1])
_H3 = network.add_shape(inputT1)
_H4 = network.add_slice(_H3.get_output(0), [1], [1], [1])
_H5 = network.add_elementwise(_H2.get_output(0), _H4.get_output(0), trt.ElementWiseOperation.EQUAL)  # 检查两个输入张量的第 1 维度长度是否相等

_H6 = network.add_identity(_H5.get_output(0))
_H6.get_output(0).dtype = trt.bool
_HA = network.add_assertion(_H6.get_output(0), "inputT0.shape[1] != inputT1.shape[1]")

_H7 = network.add_identity(_H5.get_output(0))
_H7.get_output(0).dtype = trt.int32
#-------------------------------------------------------------------------------# 网络部分
network.mark_output(_H7.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
print("\nSucceeded building engine!\n")

context = engine.create_execution_context()
context.set_binding_shape(0, data0.shape)

print("Using data0 %s <-> data1 %s" % (str(data0.shape), str(data1.shape)))
context.set_binding_shape(1, data1.shape)  # 使用 data1，可以通过 assert 检查

nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data0)
bufferH.append(data1)

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

print("Using data0 %s <-> data2 %s" % (str(data0.shape), str(data2.shape)))
context.set_binding_shape(1, data2.shape)  # 改为使用 data2，不能通过 assert 检查

bufferH = []
bufferH.append(data0)
bufferH.append(data2)
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
