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

nOut, nCOut, hOut, wOut = 1, 3, 4, 5  # 输出张量形状 NCHW

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
#-------------------------------------------------------------------------------# 网络部分
fillLayer = network.add_fill((1, 1, 1), trt.FillOperation.RANDOM_UNIFORM)
constant0 = np.ascontiguousarray(np.array([nOut, nCOut, hOut, wOut], dtype=np.int32))
constant1 = np.ascontiguousarray(np.array([-10], dtype=np.float32))
constant2 = np.ascontiguousarray(np.array([10], dtype=np.float32))
constantLayer0 = network.add_constant(constant0.shape, trt.Weights(constant0))  # 形状张量
constantLayer1 = network.add_constant([], trt.Weights(constant1))  # 最小值标量，只接受标量
constantLayer2 = network.add_constant([], trt.Weights(constant2))  # 最大指标量，只接受标量
fillLayer.set_input(0, constantLayer0.get_output(0))
fillLayer.set_input(1, constantLayer1.get_output(0))
fillLayer.set_input(2, constantLayer2.get_output(0))
#-------------------------------------------------------------------------------# 网络部分
network.mark_output(fillLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
_, stream = cudart.cudaStreamCreate()

outputH0 = np.empty(context.get_binding_shape(0), dtype=trt.nptype(engine.get_binding_dtype(0)))
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

context.execute_async_v2([int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(outputD0)