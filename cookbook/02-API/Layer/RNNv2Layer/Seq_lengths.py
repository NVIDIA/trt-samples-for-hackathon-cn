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

nB, nC, nH, nW = 1, 3, 4, 7
nHidden = 5
data = np.ones(nC * nH * nW, dtype=np.float32).reshape(nC, nH, nW)
seqLen = np.array([4, 3, 2], dtype=np.int32).reshape(nB, nC)  # 每个输入的真实长度
weightX = np.ascontiguousarray(np.ones((nHidden, nW), dtype=np.float32))
weightH = np.ascontiguousarray(np.ones((nHidden, nHidden), dtype=np.float32))
biasX = np.ascontiguousarray(np.zeros(nHidden, dtype=np.float32))
biasH = np.ascontiguousarray(np.zeros(nHidden, dtype=np.float32))

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
inputT1 = network.add_input("inputT1", trt.int32, (nB, nC))
#------------------------------------------------------------------------------- Network
rnnV2Layer = network.add_rnn_v2(inputT0, 1, nHidden, nH, trt.RNNOperation.RELU)
rnnV2Layer.seq_lengths = inputT1  # 设置每个独立输入的真实长度，默认均为 nH
rnnV2Layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(weightX))
rnnV2Layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(weightH))
rnnV2Layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(biasX))
rnnV2Layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(biasH))
#------------------------------------------------------------------------------- Network
network.mark_output(rnnV2Layer.get_output(0))
network.mark_output(rnnV2Layer.get_output(1))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data)
bufferH.append(seqLen)
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