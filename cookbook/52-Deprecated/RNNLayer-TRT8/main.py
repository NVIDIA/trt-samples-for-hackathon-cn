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
weight = np.ascontiguousarray(np.ones((nHidden, nW + nHidden), dtype=np.float32))  # 权重矩阵，X 和 H 连接在一起
bias = np.ascontiguousarray(np.zeros(nHidden * 2, dtype=np.float32))  # 偏置，bX 和 bH 连接在一起

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()  # 必须使用 implicit batch 模式
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
inputT0 = network.add_input("inputT0", trt.float32, (nC, nH, nW))
#------------------------------------------------------------------------------- Network
shuffleLayer = network.add_shuffle(inputT0)  # 先 shuffle 成 (nH,nC,nW)
shuffleLayer.first_transpose = (1, 0, 2)
fakeWeight = trt.Weights(np.random.rand(nHidden, nW + nHidden).astype(np.float32))
fakeBias = trt.Weights(np.random.rand(nHidden * 2).astype(np.float32))
rnnLayer = network.add_rnn(shuffleLayer.get_output(0), 1, nHidden, nH, trt.RNNOperation.RELU, trt.RNNInputMode.LINEAR, trt.RNNDirection.UNIDIRECTION, fakeWeight, fakeBias)
rnnLayer.weights = trt.Weights(weight)  # 重设 RNN 权重
rnnLayer.bias = trt.Weights(bias)  # 重设 RNN 偏置
#------------------------------------------------------------------------------- Network
network.mark_output(rnnLayer.get_output(0))
network.mark_output(rnnLayer.get_output(1))
engine = builder.build_engine(network, config)
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
context.execute(nB, bufferD)
for i in range(nOutput):
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput):
    print("Input %d:" % i, bufferH[i].shape, "\n", bufferH[i])
for i in range(nOutput):
    print("Output %d:" % i, bufferH[nInput + i].shape, "\n", bufferH[nInput + i])

for buffer in bufferD:
    cudart.cudaFree(buffer)
