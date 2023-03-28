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

nB, nC, nH, nW = 1, 3, 4, 5
data = np.arange(nC, dtype=np.float32).reshape(nC, 1, 1) * 100 + np.arange(nH).reshape(1, nH, 1) * 10 + np.arange(nW).reshape(1, 1, nW)
data = data.reshape(nB, nC, nH, nW).astype(np.float32)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, -1, -1))
profile.set_shape(inputT0.name, [1, 1, 1, 1], [nB, nC, nH, nW], [nB * 2, nC * 2, nH * 2, nW * 2])
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
oneLayer = network.add_constant([4], np.array([0, 1, 1, 1], dtype=np.int32))
shape0Layer = network.add_shape(inputT0)
shape1Layer = network.add_elementwise(shape0Layer.get_output(0), oneLayer.get_output(0), trt.ElementWiseOperation.SUB)
sliceLayer = network.add_slice(inputT0, (0, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))  # 给 inputT0 除了最高维以外每一维减少 1
sliceLayer.set_input(2, shape1Layer.get_output(0))
#shape1 = [1] + [ x-1 for x in inputT0.shape[1:] ]
#sliceLayer = network.add_slice(inputT0,(0,0,0,0),shape1,(1,1,1,1))  # 错误的做法，因为 dynamic shape 模式下 inputT0.shape 可能含有 -1，不能作为新形状
#------------------------------------------------------------------------------- Network
network.mark_output(sliceLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], data.shape)

bufferH = []
bufferH.append(data)
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