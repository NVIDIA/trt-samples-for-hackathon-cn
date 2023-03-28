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

nOut, nCOut, hOut, wOut = 1, 3, 4, 5
data0 = np.array([nOut, nCOut, hOut, wOut], dtype=np.int32)
data1 = np.float32(1000).reshape(-1)
data2 = np.array([0, 100, 10, 1], dtype=np.float32)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.int32, [4])
inputT1 = network.add_input("inputT1", trt.float32, [])
inputT2 = network.add_input("inputT2", trt.float32, [4])
profile.set_shape_input(inputT0.name, (1, 1, 1, 1), (nOut, nCOut, hOut, wOut), (5, 5, 5, 5))  # range of value rather than shape
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
fillLayer = network.add_fill([-1], trt.FillOperation.LINSPACE)
fillLayer.set_input(0, inputT0)
fillLayer.set_input(1, inputT1)
fillLayer.set_input(2, inputT2)
#------------------------------------------------------------------------------- Network
network.mark_output(fillLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_tensor_address("inputT0", np.array([nOut, nCOut, hOut, wOut], dtype=np.int32).ctypes.data)
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(data0))
bufferH.append(np.ascontiguousarray(data1))
bufferH.append(np.ascontiguousarray(data2))
for i in range(nInput, nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))  # get_tensor_shape() will give (-1,-1,-1,-1) for output shape
bufferD = []
bufferD.append(bufferH[0].ctypes.data)  # the input shape tensor is on host
for i in range(1, nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(1, nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(1, nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)