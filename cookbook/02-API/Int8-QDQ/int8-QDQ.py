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

import numpy as np
from cuda import cudart
import tensorrt as trt

nIn, cIn, hIn, wIn = 1, 1, 6, 9
cOut, hW, wW = 1, 3, 3
data = np.tile(np.arange(1, 1 + hW * wW, dtype=np.float32).reshape(hW, wW), (cIn, hIn // hW, wIn // wW)).reshape(1, cIn, hIn, wIn)
weight = np.full([cOut, hW, wW], 1, dtype=np.float32)
bias = np.zeros(cOut, dtype=np.float32)

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.flags = 1 << int(trt.BuilderFlag.INT8)  # 需要打开 int8 模式
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn, cIn, hIn, wIn))

qValue = 1/1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
inputQLayer = network.add_quantize(inputT0, qTensor)
inputQLayer.axis = 0
inputQDQLayer = network.add_dequantize(inputQLayer.get_output(0), qTensor)
inputQDQLayer.axis = 0

weightLayer = network.add_constant([cOut, cIn, hW, wW], weight)
qValue = 1/1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
weightQLayer = network.add_quantize(weightLayer.get_output(0), qTensor)
weightQLayer.axis = 0
weightQDQLayer = network.add_dequantize(weightQLayer.get_output(0), qTensor)
weightQDQLayer.axis = 0

convolutionLayer = network.add_convolution_nd(inputQDQLayer.get_output(0), cOut, (hW, wW), trt.Weights())  # 需要把 weight 设为空权重（不能用 np.array()）
convolutionLayer.padding_nd = [hW//2,wW//2]
convolutionLayer.set_input(1, weightQDQLayer.get_output(0))

qValue = 1/1
qTensor = network.add_constant([], np.array([qValue], dtype=np.float32)).get_output(0)
convQLayer = network.add_quantize(convolutionLayer.get_output(0), qTensor)
convQLayer.axis = 0
convQDQLayer = network.add_dequantize(convQLayer.get_output(0), qTensor)
convQDQLayer.axis = 0

network.mark_output(convQDQLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
_, stream = cudart.cudaStreamCreate()

inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
