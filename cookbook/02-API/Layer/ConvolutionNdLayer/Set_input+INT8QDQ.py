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

nB, nC, nH, nW = 1, 1, 6, 9
nCOut, nKernelHeight, nKernelWidth = 1, 3, 3
data = np.tile(np.arange(1, 1 + nKernelHeight * nKernelWidth, dtype=np.float32).reshape(nKernelHeight, nKernelWidth), (nC, nH // nKernelHeight, nW // nKernelWidth)).reshape(1, nC, nH, nW)
weight = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(nCOut, nKernelHeight, nKernelWidth))
bias = np.ascontiguousarray(np.zeros(nCOut, dtype=np.float32))

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)  # need INT8 mode
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
#------------------------------------------------------------------------------- Network
constantLayer0 = network.add_constant([], np.array([1], dtype=np.float32))
constantLayer1 = network.add_constant([], np.array([1], dtype=np.float32))
weightLayer = network.add_constant([nCOut, nC, nKernelHeight, nKernelWidth], weight)

quantizeLayer0 = network.add_quantize(inputT0, constantLayer0.get_output(0))
quantizeLayer0.axis = 0
dequantizeLayer0 = network.add_dequantize(quantizeLayer0.get_output(0), constantLayer1.get_output(0))
dequantizeLayer0.axis = 0
quantizeLayer1 = network.add_quantize(weightLayer.get_output(0), constantLayer0.get_output(0))
quantizeLayer1.axis = 0
dequantizeLayer1 = network.add_dequantize(quantizeLayer1.get_output(0), constantLayer1.get_output(0))
dequantizeLayer1.axis = 0

convolutionLayer = network.add_convolution_nd(dequantizeLayer0.get_output(0), nCOut, (nKernelHeight, nKernelWidth), trt.Weights(), trt.Weights(bias))  # set weight as empty in the constructor
convolutionLayer.set_input(1, dequantizeLayer1.get_output(0))
#------------------------------------------------------------------------------- Network
network.mark_output(convolutionLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(data))
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