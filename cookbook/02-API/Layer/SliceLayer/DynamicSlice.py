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
data0 = np.arange(nC, dtype=np.float32).reshape(nC, 1, 1) * 100 + np.arange(nH).reshape(1, nH, 1) * 10 + np.arange(nW).reshape(1, 1, nW)
data0 = data0.reshape(nB, nC, nH, nW).astype(np.float32)
data1 = np.array([0, 0, 0, 0], dtype=np.int32)
data2 = np.array([1, 2, 3, 4], dtype=np.int32)
data3 = np.array([1, 1, 1, 1], dtype=np.int32)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()  # 需要使用 profile
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
inputT1 = network.add_input("inputT1", trt.int32, (4, ))
inputT2 = network.add_input("inputT2", trt.int32, (4, ))
inputT3 = network.add_input("inputT3", trt.int32, (4, ))
profile.set_shape_input(inputT1.name, (0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2))  # 这里设置的不是 shape input 的形状而是值
profile.set_shape_input(inputT2.name, (1, 1, 1, 1), (1, 2, 3, 4), (1, 3, 4, 5))
profile.set_shape_input(inputT3.name, (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
sliceLayer = network.add_slice(inputT0, (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
#sliceLayer.set_input(0,inputT0)
sliceLayer.set_input(1, inputT1)
sliceLayer.set_input(2, inputT2)
sliceLayer.set_input(3, inputT3)
#------------------------------------------------------------------------------- Network
network.mark_output(sliceLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_shape_input(1, data1)  # 运行时绑定真实形状张量值
context.set_shape_input(2, data2)
context.set_shape_input(3, data3)
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data0)
bufferH.append(data1)
bufferH.append(data2)
bufferH.append(data3)
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