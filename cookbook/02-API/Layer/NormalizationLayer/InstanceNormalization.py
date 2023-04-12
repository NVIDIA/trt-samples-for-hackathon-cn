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
import tensorrt as trt
from cuda import cudart

np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

shape = [1, 3, 4, 5]
shapeSB = [1] + shape[1:2] + [1, 1]  # shape[1,3,1,1]
data0 = np.arange(np.prod(shape[2:]), dtype=np.float32).reshape(1, 1, *shape[2:])
data1 = 100 - np.arange(np.prod(shape[2:]), dtype=np.float32).reshape(1, 1, *shape[2:])
data2 = np.ones(shape[2:], dtype=np.float32).reshape(1, 1, *shape[2:])
data = np.concatenate([data0, data1, data2], axis=1)
scale = np.full(shapeSB, 1, dtype=np.float32)
bias = np.full(shapeSB, 0, dtype=np.float32)

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, shape)
inputT1 = network.add_input("inputT1", trt.float32, shapeSB)
inputT2 = network.add_input("inputT2", trt.float32, shapeSB)
#------------------------------------------------------------------------------- Network
normalizationLayer = network.add_normalization(inputT0, inputT1, inputT2, 1 << 2 | 1 << 3)
#normalizationLayer.epsilon = 1  # set epsilon, we can set it as 1 to compare the differnece of the output
#normalizationLayer.axes = 1 << 3  # reset the axes mask after the constructor
#normalizationLayer.compute_precision = trt.float32  # set the precision of computation
#------------------------------------------------------------------------------- Network
network.mark_output(normalizationLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(data)
bufferH.append(scale)
bufferH.append(bias)
for i in range(nIO):
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