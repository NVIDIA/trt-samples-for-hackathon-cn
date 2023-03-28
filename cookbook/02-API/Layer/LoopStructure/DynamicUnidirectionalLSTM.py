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
x = np.ones([nBatchSize, nSequenceLength, nInputDim], dtype=np.float32)
h0 = np.ones([nBatchSize, nHiddenDim], dtype=np.float32)  # initial hidden state
c0 = np.zeros([nBatchSize, nHiddenDim], dtype=np.float32)  # initial cell state
weightAllX = np.ones((nHiddenDim, nInputDim), dtype=np.float32)  # weight of X->H
weightAllH = np.ones((nHiddenDim, nHiddenDim), dtype=np.float32)  # weight of H->H
biasAllX = np.zeros(nHiddenDim, dtype=np.float32)  # bias of X->H
biasAllH = np.zeros(nHiddenDim, dtype=np.float32)  # bias of H->H

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))  # 3 inputs of x, h0, c0
inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
config.add_optimization_profile(profile)

#------------------------------------------------------------------------------- Network
def gate(network, xTensor, wx, hTensor, wh, b, isSigmoid):
    _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
    _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
    _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
    _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
    _h4 = network.add_activation(_h3.get_output(0), [trt.ActivationType.TANH, trt.ActivationType.SIGMOID][int(isSigmoid)])
    return _h4

weightAllXLayer = network.add_constant([nInputDim, nHiddenDim], trt.Weights(np.ascontiguousarray(weightAllX.transpose())))
weightAllHLayer = network.add_constant([nHiddenDim, nHiddenDim], trt.Weights(np.ascontiguousarray(weightAllH.transpose())))
biasAllLayer = network.add_constant([1, nHiddenDim], trt.Weights(np.ascontiguousarray(biasAllX + biasAllH)))

_t0 = network.add_shape(inputT0)
_t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
_t2 = network.add_shuffle(_t1.get_output(0))  # scalar tensor needed for the two kinds of loop condition
_t2.reshape_dims = ()

loop = network.add_loop()
loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
iteratorLayer = loop.add_iterator(inputT0, 1, False)  # one piece of inputT0 [nBatchSize, nInputDim] is indexed for computation each time, an additional anti-throw iterator is needed for two-way LSTM

hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state. There are multiple loop variables in a loop.
cellStateLayer = loop.add_recurrence(inputT2)

gateI = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateF = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateC = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), False)
gateO = gate(network, iteratorLayer.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)

_h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
_h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
_h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # shape [nBatchSize,nHiddenSize]
loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # shape [nBatchSize,nSequenceLength,nHiddenSize]
loopOutput1.set_input(1, _t2.get_output(0))
loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # shape [nBatchSize,nHiddenSize]
#------------------------------------------------------------------------------- Network
network.mark_output(loopOutput0.get_output(0))
network.mark_output(loopOutput1.get_output(0))
network.mark_output(loopOutput2.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], x.shape)
context.set_input_shape(lTensorName[1], h0.shape)
context.set_input_shape(lTensorName[2], c0.shape)

bufferH = []
bufferH.append(x)
bufferH.append(h0)
bufferH.append(c0)
for i in range(nInput, nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)