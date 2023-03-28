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

nBatchSize, nSequenceLength, nInputDim = 3, 4, 7
nHiddenDim = 5
data = np.ones([nBatchSize, nSequenceLength, nInputDim], dtype=np.float32)
weightAllX = np.ones((nHiddenDim, nInputDim), dtype=np.float32)  # 权重矩阵 (X->H)
weightAllH = np.ones((nHiddenDim, nHiddenDim), dtype=np.float32)  # 权重矩阵 (H->H)
biasAllX = np.zeros(nHiddenDim, dtype=np.float32)  # 偏置 (X->H)
biasAllH = np.zeros(nHiddenDim, dtype=np.float32)  # 偏置 (H->H)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nBatchSize, nSequenceLength, nInputDim))  # 采用单输入网络

#-------------------------------------------------------------------------------
def gate(network, x, wx, hiddenStateLayer, wh, b, isSigmoid):
    _h0 = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
    _h1 = network.add_matrix_multiply(hiddenStateLayer, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
    _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
    _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
    _h4 = network.add_activation(_h3.get_output(0), [trt.ActivationType.TANH, trt.ActivationType.SIGMOID][int(isSigmoid)])
    return _h4

weightAllXLayer = network.add_constant([nInputDim, nHiddenDim], trt.Weights(np.ascontiguousarray(weightAllX.transpose())))
weightAllHLayer = network.add_constant([nHiddenDim, nHiddenDim], trt.Weights(np.ascontiguousarray(weightAllH.transpose())))
biasAllLayer = network.add_constant([1, nHiddenDim], trt.Weights(np.ascontiguousarray(biasAllX + biasAllH)))
hidden0Layer = network.add_constant([nBatchSize, nHiddenDim], trt.Weights(np.ascontiguousarray(np.ones(nBatchSize * nHiddenDim, dtype=np.float32))))  # 初始隐藏状态
cell0Layer = network.add_constant([nBatchSize, nHiddenDim], trt.Weights(np.ascontiguousarray(np.zeros(nBatchSize * nHiddenDim, dtype=np.float32))))  # 初始细胞状态
length = network.add_constant((), np.array([nSequenceLength], dtype=np.int32))

loop = network.add_loop()
loop.add_trip_limit(length.get_output(0), trt.TripLimit.COUNT)
iteratorLayer = loop.add_iterator(inputT0, 1, False)  # 每次抛出 inputTensor 的 H 维的一层 (nBatchSize,nInputDim)，双向 LSTM 要多一个反抛的迭代器
hiddenStateLayer = loop.add_recurrence(hidden0Layer.get_output(0))  # 一个 loop 中有多个循环变量
cellStateLayer = loop.add_recurrence(cell0Layer.get_output(0))

gateI = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateC = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), False)
gateF = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateO = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)

_h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
_h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
_h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # 形状 (nBatchSize,nHiddenDim)，nBatchSize 个独立输出，每个隐藏状态 nHiddenDim 维坐标
loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # 形状 (nSequenceLength,nBatchSize,nHiddenDim)，nBatchSize 个独立输出，每个输出 nSequenceLength 个隐藏状态，每个隐藏状态 nHiddenDim 维坐标
loopOutput1.set_input(1, length.get_output(0))
loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # 形状 (nBatchSize,nHiddenDim)，nBatchSize 个独立输出，每个隐藏状态 nHiddenDim 维坐标
#-------------------------------------------------------------------------------
network.mark_output(loopOutput0.get_output(0))
network.mark_output(loopOutput1.get_output(0))
network.mark_output(loopOutput2.get_output(0))
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