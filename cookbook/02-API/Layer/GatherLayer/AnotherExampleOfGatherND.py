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

from cuda import cudart
import numpy as np
import tensorrt as trt

data0 = np.arange(3 * 3 * 4 * 5, dtype=np.float32).reshape(3, 3, 4, 5)
data1 = np.zeros([3, 1, 4], dtype=np.int32) + 0
data2 = np.zeros([3, 1, 4], dtype=np.int32) + 1
data3 = np.zeros([3, 1, 4], dtype=np.int32) + 2

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
profile.set_shape(inputT0.name, [1, 3, 4, 5], [3, 3, 4, 5], [6, 6, 8, 10])
indexT0 = network.add_input("indexT0", trt.int32, [-1, 1, 4])
profile.set_shape(indexT0.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
indexT1 = network.add_input("indexT1", trt.int32, [-1, 1, 4])
profile.set_shape(indexT1.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
indexT2 = network.add_input("indexT2", trt.int32, [-1, 1, 4])
profile.set_shape(indexT2.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
indexL = network.add_concatenation([indexT0, indexT1, indexT2])
indexL.axis = 1

indexTL = network.add_shuffle(indexL.get_output(0))
indexTL.first_transpose = [0, 2, 1]

outputL = network.add_gather(inputT0, indexTL.get_output(0), 1)
outputL.mode = trt.GatherMode.ND
outputL.num_elementwise_dims = 1
#------------------------------------------------------------------------------- Network
network.mark_output(outputL.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [3, 3, 4, 5])
context.set_input_shape(lTensorName[1], [3, 1, 4])
context.set_input_shape(lTensorName[2], [3, 1, 4])
context.set_input_shape(lTensorName[3], [3, 1, 4])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(data0))
bufferH.append(np.ascontiguousarray(data1))
bufferH.append(np.ascontiguousarray(data2))
bufferH.append(np.ascontiguousarray(data3))
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
