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

nB, nC, nH, nW = 2, 3, 4, 5
data0 = np.arange(nB * nC * nH * nW, dtype=np.float32).reshape(nB, nC, nH, nW)
data1 = np.array([[[0, 2, 1, 1], [1, 0, 3, 2], [0, 1, 2, 3]], [[1, 2, 1, 1], [0, 0, 3, 2], [1, 1, 2, 3]]], dtype=np.int32)
data2 = -np.arange(nB * nC, dtype=np.float32).reshape(nB, nC)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

def scatterCPU(data0, data1, data2):
    output = data0
    for i in range(2):
        for j in range(3):
            #print("i=%d,j=%d,index=%s,updateValue=%f" % (i, j, data1[i, j], data2[i, j]))
            output[data1[i, j][0], data1[i, j][1], data1[i, j][2], data1[i, j][3]] = data2[i, j]
    return output

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, [nB, nC, nH, nW])
inputT1 = network.add_input("inputT1", trt.int32, [nB, nC, nH])
inputT2 = network.add_input("inputT2", trt.float32, [nB, nC])
#------------------------------------------------------------------------------- Network
scatterLayer = network.add_scatter(inputT0, inputT1, inputT2, trt.ScatterMode.ND)
#------------------------------------------------------------------------------- Network
network.mark_output(scatterLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data0)
bufferH.append(data1)
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

resCPU = scatterCPU(data0, data1, data2)
print("diff:\n", bufferH[nInput] - resCPU)