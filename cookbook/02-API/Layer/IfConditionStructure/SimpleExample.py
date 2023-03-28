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
data0 = np.arange(1, 1 + nB * nC * nH * nW, dtype=np.float32).reshape(nB, nC, nH, nW)
data1 = data0 - 1

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
#------------------------------------------------------------------------------- Network
# condition: inputT0.reshape(-1)[0] != 0
_H0 = network.add_slice(inputT0, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
_H1 = network.add_reduce(_H0.get_output(0), trt.ReduceOperation.SUM, (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3), False)
_H2 = network.add_identity(_H1.get_output(0))
_H2.set_output_type(0, trt.bool)
_H2.get_output(0).dtype = trt.bool

ifCondition = network.add_if_conditional()
ifConditionInputLayer = ifCondition.add_input(inputT0)
ifConditionConditionLayer = ifCondition.set_condition(_H2.get_output(0))  # the condition tensor must be BOOL data type with 0 dimension

# branch of condition is true
_H3 = network.add_elementwise(ifConditionInputLayer.get_output(0), ifConditionInputLayer.get_output(0), trt.ElementWiseOperation.SUM)

# branch of condition is false
_H4 = network.add_identity(ifConditionInputLayer.get_output(0))

ifConditionOutputLayer = ifCondition.add_output(_H3.get_output(0), _H4.get_output(0))
#------------------------------------------------------------------------------- Network
network.mark_output(ifConditionOutputLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

bufferH[0] = data0  # input data satisfies the condition

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

bufferH[0] = data1  # input data does not satisfy the condition

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

for buffer in bufferD:
    cudart.cudaFree(buffer)