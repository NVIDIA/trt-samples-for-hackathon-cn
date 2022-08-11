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

nB, nC, nH, nW = 1, 1, 6, 9
nCOut, nKernelHeight, nKernelWidth = 1, 3, 3
data = np.tile(np.arange(1, 1 + nKernelHeight * nKernelWidth, dtype=np.float32).reshape(nKernelHeight, nKernelWidth), (nC, nH // nKernelHeight, nW // nKernelWidth)).reshape(1, nC, nH, nW)
weight = np.ascontiguousarray(np.full([nCOut, nKernelHeight, nKernelWidth], 1, dtype=np.float32))

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.flags = 1 << int(trt.BuilderFlag.INT8)  # 需要打开 int8 模式
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))

qValue = 1 / 1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
inputQLayer = network.add_quantize(inputT0, qTensor)
inputQLayer.axis = 0
inputQDQLayer = network.add_dequantize(inputQLayer.get_output(0), qTensor)
inputQDQLayer.axis = 0

weightLayer = network.add_constant([nCOut, nC, nKernelHeight, nKernelWidth], trt.Weights(weight))
qValue = 1 / 1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
weightQLayer = network.add_quantize(weightLayer.get_output(0), qTensor)
weightQLayer.axis = 0
weightQDQLayer = network.add_dequantize(weightQLayer.get_output(0), qTensor)
weightQDQLayer.axis = 0

convolutionLayer = network.add_convolution_nd(inputQDQLayer.get_output(0), nCOut, (nKernelHeight, nKernelWidth), trt.Weights())  # 需要把 weight 设为空权重（不能用 np.array()）
convolutionLayer.padding_nd = [nKernelHeight // 2, nKernelWidth // 2]
convolutionLayer.set_input(1, weightQDQLayer.get_output(0))

qValue = 1 / 1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
convQLayer = network.add_quantize(convolutionLayer.get_output(0), qTensor)
convQLayer.axis = 0
convQDQLayer = network.add_dequantize(convQLayer.get_output(0), qTensor)
convQDQLayer.axis = 0

network.mark_output(convQDQLayer.get_output(0))
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