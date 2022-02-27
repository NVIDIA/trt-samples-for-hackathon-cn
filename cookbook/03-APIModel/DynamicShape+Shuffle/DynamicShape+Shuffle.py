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
import tensorrt as trt
from cuda import cudart

cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputTensor = network.add_input('inputT0', trt.DataType.FLOAT, [-1, -1, -1])  # 只想对末尾的一维施用全连接
profile.set_shape(inputTensor.name, (1, 1, 1), (3, 4, 5), (6, 8, 10))
config.add_optimization_profile(profile)

oneByOneH = network.add_constant((2, ), np.ascontiguousarray(np.array([1, 1], dtype=np.int32)))

_H1 = network.add_shape(inputTensor)  # 在原张量末尾添加额外两维 (1,1)
_H2 = network.add_concatenation([_H1.get_output(0), oneByOneH.get_output(0)])
_H3 = network.add_shuffle(inputTensor)
_H3.set_input(1, _H2.get_output(0))
#_H3         = network.add_shuffle(inputTensor)  # 错误的做法，dynamic shape 模式中 .shape 得到的形状中可能含有多个 -1，不可作为 shuffle 的新形状
#_H3.reshape_dims = tuple(inputTensor.shape)+(1,1)

weight = np.ones(5, dtype=np.float32)
weight = np.concatenate([weight, -weight], 0)
bias = np.full(2, 0.5, dtype=np.float32)
FCH = network.add_fully_connected(_H3.get_output(0), 2, weight, bias)

_H4 = network.add_shape(FCH.get_output(0))  # 去掉结果张量末尾的额外两维 (1,1)
_H5 = network.add_slice(_H4.get_output(0), [0], [_H4.get_output(0).shape[0] - 2], [1])  # 也可直接用 _H5 = _H1
_H6 = network.add_shuffle(FCH.get_output(0))
_H6.set_input(1, _H5.get_output(0))
#_H6         = network.add_shuffle(FCH.get_output(0))                                               # 错误的做法，理由与前面类似
#_H6.reshape_dims = tuple(FCH.get_output(0).shape[:-2])

network.mark_output(_H6.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_binding_shape(0, [3, 4, 5])
_, stream = cudart.cudaStreamCreate()

data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0:", data.shape)
print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
