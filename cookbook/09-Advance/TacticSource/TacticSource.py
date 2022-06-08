#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
import numpy as np
from time import time
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"
nIn, cIn, hIn, wIn = 1, 1, 28, 28
np.random.seed(97)
data = np.random.rand(nIn, cIn, hIn, wIn).astype(np.float32) * 2 - 1

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 6 << 30
config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
#config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT))

inputTensor = network.add_input('inputT0', trt.float32, [-1, 1, 28, 28])
profile.set_shape(inputTensor.name, [1, cIn, hIn, wIn], [nIn, cIn, hIn, wIn], [nIn * 2, cIn, hIn, wIn])
config.add_optimization_profile(profile)

w = np.random.rand(32, 1, 5, 5).astype(np.float32).reshape(-1)
b = np.random.rand(32).astype(np.float32).reshape(-1)
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.random.rand(64, 32, 5, 5).astype(np.float32).reshape(-1)
b = np.random.rand(64).astype(np.float32).reshape(-1)
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.first_transpose = (0, 2, 3, 1)
_6.reshape_dims = (-1, 64 * 7 * 7, 1, 1)

w = np.random.rand(1024, 64 * 7 * 7).astype(np.float32).reshape(-1)
b = np.random.rand(1024).astype(np.float32).reshape(-1)
_7 = network.add_fully_connected(_6.get_output(0), 1024, w, b)
_8 = network.add_activation(_7.get_output(0), trt.ActivationType.RELU)

w = np.random.rand(10, 1024).astype(np.float32).reshape(-1)
b = np.random.rand(10).astype(np.float32).reshape(-1)
_9 = network.add_fully_connected(_8.get_output(0), 10, w, b)
_10 = network.add_activation(_9.get_output(0), trt.ActivationType.RELU)

_11 = network.add_shuffle(_10.get_output(0))
_11.reshape_dims = [-1, 10]

_12 = network.add_softmax(_11.get_output(0))
_12.axes = 1 << 1

_13 = network.add_topk(_12.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

network.mark_output(_13.get_output(1))

engineString = builder.build_serialized_network(network, config)

if engineString == None:
    print("Failed getting serialized engine!")
    exit()
print("Succeeded getting serialized engine!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
if engine == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")

context = engine.create_execution_context()
context.set_binding_shape(0, [nIn, cIn, hIn, wIn])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput + nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

bufferH = []
bufferH.append(np.ascontiguousarray(data.reshape(-1)))
for i in range(nInput, nInput + nOutput):
    bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
bufferD = []
for i in range(nInput + nOutput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput + nOutput):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

t0 = time()
for i in range(10):
    context.execute_v2(bufferD)
t1 = time()
print("Timing:%f ms" % ((t1 - t0) * 1000))

for i in range(nInput + nOutput):
    print(engine.get_binding_name(i))

for b in bufferD:
    cudart.cudaFree(b)
