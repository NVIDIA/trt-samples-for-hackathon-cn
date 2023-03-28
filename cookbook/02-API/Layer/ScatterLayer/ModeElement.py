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

np.random.seed(31193)
nB, nC, nH, nW = 1, 3, 4, 5
data0 = np.arange(nB * nC * nH * nW, dtype=np.float32).reshape(nB, nC, nH, nW)
data1 = np.tile(np.arange(nH), [nB, nC, 1, nW]).astype(np.int32).reshape(nB, nC, nH, nW)
data2 = -data0
scatterAxis = 2

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

def scatterCPU(data0, data1, data2, axis):  # 用于说明算法
    nB, nC, nH, nW = data0.shape
    output = data0
    if axis == 0:
        for n in range(nB):
            for c in range(nC):
                for h in range(nH):
                    for w in range(nW):
                        output[data1[n, c, h, w], c, h, w] = data2[n, c, h, w]
                        #print("<%d,%d,%d,%d>[%d,%d,%d,%d],%f>"%(n,c,h,w,n,c,data1[n,c,h,w],w,data2[n,c,h,w]))
                        #print(output)
    elif axis == 1:
        for n in range(nB):
            for c in range(nC):
                for h in range(nH):
                    for w in range(nW):
                        output[n, data1[n, c, h, w], h, w] = data2[n, c, h, w]
                        #print("<%d,%d,%d,%d>[%d,%d,%d,%d],%f>"%(n,c,h,w,n,c,data1[n,c,h,w],w,data2[n,c,h,w]))
                        #print(output)
    elif axis == 2:
        for n in range(nB):
            for c in range(nC):
                for h in range(nH):
                    for w in range(nW):
                        output[n, c, data1[n, c, h, w], w] = data2[n, c, h, w]
                        #print("<%d,%d,%d,%d>[%d,%d,%d,%d],%f>"%(n,c,h,w,n,c,data1[n,c,h,w],w,data2[n,c,h,w]))
                        #print(output)
    elif axis == 3:
        for n in range(nB):
            for c in range(nC):
                for h in range(nH):
                    for w in range(nW):
                        output[n, c, h, data1[n, c, h, w]] = data2[n, c, h, w]
                        #print("<%d,%d,%d,%d>[%d,%d,%d,%d],%f>"%(n,c,h,w,n,c,data1[n,c,h,w],w,data2[n,c,h,w]))
                        #print(output)
    else:
        print("Failed scattering at axis %d !" % axis)
        return None
    return output

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
inputT1 = network.add_input("inputT1", trt.int32, (nB, nC, nH, nW))
inputT2 = network.add_input("inputT2", trt.float32, (nB, nC, nH, nW))
#------------------------------------------------------------------------------- Network
scatterLayer = network.add_scatter(inputT0, inputT1, inputT2, trt.ScatterMode.ELEMENT)
scatterLayer.axis = scatterAxis
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

resCPU = scatterCPU(data0, data1, data2, scatterAxis)
print("diff:\n", bufferH[nInput] - resCPU)