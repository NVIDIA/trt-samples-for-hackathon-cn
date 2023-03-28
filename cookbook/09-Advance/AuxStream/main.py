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
import nvtx
import tensorrt as trt
from cuda import cudart

np.random.seed(31193)

nGEMM = 10
nMKN = 128

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

config.max_aux_streams = 2

inputList = []
for i in range(nGEMM + 1):
    inputT = network.add_input("inputT" + str(i), trt.float32, [-1, 4, nMKN, nMKN])
    profile.set_shape(inputT.name, [1, 4, nMKN, nMKN], [4, 4, nMKN, nMKN], [8, 4, nMKN, nMKN])
    inputList.append(inputT)
config.add_optimization_profile(profile)

tempTensor0 = inputList[0]
tempTensor1 = inputList[0]
for i in range(1, nGEMM + 1):
    tempLayer0 = network.add_matrix_multiply(tempTensor0, trt.MatrixOperation.NONE, inputList[i], trt.MatrixOperation.NONE)
    tempTensor0 = tempLayer0.get_output(0)

    tempLayer1 = network.add_matrix_multiply(tempTensor1, trt.MatrixOperation.NONE, inputList[nGEMM + 1 - i], trt.MatrixOperation.NONE)
    tempTensor1 = tempLayer1.get_output(0)

network.mark_output(tempTensor0)
network.mark_output(tempTensor1)

engineString = builder.build_serialized_network(network, config)

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

#print("Engine.max_aux_streams=%d" % engine.max_aux_streams)

context = engine.create_execution_context()
for i in range(nInput):
    context.set_input_shape(lTensorName[i], [4, 4, nMKN, nMKN])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

context.set_aux_streams([cudart.cudaStreamCreate()[1] for i in range(max(1, engine.num_aux_streams))])  # optional, TensorRT wiil create residual cudaStream besides we assign to context

bufferH = []
for i in range(nInput):
    bufferH.append(np.ascontiguousarray(np.random.rand(np.prod([4, 4, nMKN, nMKN]))).astype(np.float32).reshape(4, 4, nMKN, nMKN) * 2 - 1)
for i in range(nInput, nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

# warm up
context.execute_async_v3(0)

with nvtx.annotate("Inference", color="green"):
    context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(np.sum(bufferH[i]))

for b in bufferD:
    cudart.cudaFree(b)
