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

np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

nB, nC, nH0, nW0 = 1, 3, 4, 5
nH1, nW1 = 6, 10
data0 = np.arange(nB).reshape(nB, 1, 1, 1) * 1000 + np.arange(nC).reshape(1, nC, 1, 1) * 100 + np.arange(nH0).reshape(1, 1, nH0, 1) * 10 + np.arange(nW0).reshape(1, 1, 1, nW0)
data0 = data0.astype(np.float32)
dataX = np.random.randint(0, nH0, [nB, nH1, nW1, 1], dtype=np.int32) / (nH0 - 1) * 2 - 1
dataY = np.random.randint(0, nW0, [nB, nH1, nW1, 1], dtype=np.int32) / (nW0 - 1) * 2 - 1
data1 = np.concatenate([dataX, dataY], axis=3).astype(np.float32)

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH0, nW0))
inputT1 = network.add_input("inputT1", trt.float32, (nB, nH1, nW1, 2))

#------------------------------------------------------------------------------- Network
gridSampleLayer = network.add_grid_sample(inputT0, inputT1)
#gridSampleLayer.interpolation_mode = trt.InterpolationMode.NEAREST
#gridSampleLayer.interpolation_mode = trt.InterpolationMode.LINEAR  # default value
#gridSampleLayer.interpolation_mode = trt.InterpolationMode.CUBIC
#gridSampleLayer.align_corners = True
#gridSampleLayer.align_corners = False  # default value
#gridSampleLayer.sample_mode = trt.SampleMode.DEFAULT  # the same as STRICT_BOUNDS, deprecated since TensorRT 8.5
#gridSampleLayer.sample_mode = trt.SampleMode.STRICT_BOUNDS
#gridSampleLayer.sample_mode = trt.SampleMode.WRAP
#gridSampleLayer.sample_mode = trt.SampleMode.CLAMP
#gridSampleLayer.sample_mode = trt.SampleMode.FILL  # default value
#gridSampleLayer.sample_mode = trt.SampleMode.REFLECT
#------------------------------------------------------------------------------- Network
network.mark_output(gridSampleLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

bufferH[0] = data0
bufferH[1] = data1

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