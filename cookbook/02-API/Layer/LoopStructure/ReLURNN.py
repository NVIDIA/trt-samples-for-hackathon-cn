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

nBatchSize, nSequenceLength, nInputDim = 3, 4, 7  # 输入张量尺寸
nHiddenDim = 5  # 隐藏层宽度
data = np.ones([nBatchSize, nSequenceLength, nInputDim], dtype=np.float32)
weightX = np.ones((nHiddenDim, nInputDim), dtype=np.float32)  # 权重矩阵 (X->H)
weightH = np.ones((nHiddenDim, nHiddenDim), dtype=np.float32)  # 权重矩阵 (H->H)
biasX = np.zeros(nHiddenDim, dtype=np.float32)  # 偏置 (X->H)
biasH = np.zeros(nHiddenDim, dtype=np.float32)  # 偏置 (H->H)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nBatchSize, nSequenceLength, nInputDim))
#------------------------------------------------------------------------------- Network
weightXLayer = network.add_constant([nInputDim, nHiddenDim], weightX.transpose().reshape(-1))
weightHLayer = network.add_constant([nHiddenDim, nHiddenDim], weightH.transpose().reshape(-1))
biasLayer = network.add_constant([nBatchSize, nHiddenDim], np.tile(biasX + biasH, (nBatchSize, 1)))
hidden0Layer = network.add_constant([nBatchSize, nHiddenDim], np.ones(nBatchSize * nHiddenDim, dtype=np.float32))  # 初始隐藏状态，注意形状和 RNNV2层的不一样
lengthLayer = network.add_constant((), np.array([nSequenceLength], dtype=np.int32))  # 结果保留长度

loop = network.add_loop()
loop.add_trip_limit(lengthLayer.get_output(0), trt.TripLimit.COUNT)
iteratorLayer = loop.add_iterator(inputT0, 1, False)  # 每次抛出 inputTensor 的 H 维的一层 (nBatchSize,nInputDim)

rLayer = loop.add_recurrence(hidden0Layer.get_output(0))
_H0 = network.add_matrix_multiply(iteratorLayer.get_output(0), trt.MatrixOperation.NONE, weightXLayer.get_output(0), trt.MatrixOperation.NONE)
_H1 = network.add_matrix_multiply(rLayer.get_output(0), trt.MatrixOperation.NONE, weightHLayer.get_output(0), trt.MatrixOperation.NONE)
_H2 = network.add_elementwise(_H0.get_output(0), _H1.get_output(0), trt.ElementWiseOperation.SUM)
_H3 = network.add_elementwise(_H2.get_output(0), biasLayer.get_output(0), trt.ElementWiseOperation.SUM)
_H4 = network.add_activation(_H3.get_output(0), trt.ActivationType.RELU)
rLayer.set_input(1, _H4.get_output(0))

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # 形状 (nBatchSize,nHiddenDim)，nBatchSize 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 nHiddenDim 维坐标
loopOutput1 = loop.add_loop_output(_H4.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # 形状 (nSequenceLength,nBatchSize,nHiddenDim)，nBatchSize 个独立输出，每个输出 nSequenceLength 个隐藏状态，每个隐藏状态 nHiddenDim 维坐标
loopOutput1.set_input(1, lengthLayer.get_output(0))
#------------------------------------------------------------------------------- Network
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