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
data = np.ones([nB, nC, nH, nW], dtype=np.float32)
t = np.array([6], dtype=np.int32)  # number of iterations

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
inputT1 = network.add_input("inputT1", trt.int32, ())  # set number of iterations as input tensor at runtime
profile.set_shape_input(inputT1.name, (1, ), (6, ), (10, ))  # set value (rather than shape)
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
loop = network.add_loop()
loop.add_trip_limit(inputT1, trt.TripLimit.COUNT)

rLayer = loop.add_recurrence(inputT0)
_H0 = network.add_elementwise(rLayer.get_output(0), rLayer.get_output(0), trt.ElementWiseOperation.SUM)
rLayer.set_input(1, _H0.get_output(0))

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
loopOutput1 = loop.add_loop_output(_H0.get_output(0), trt.LoopOutput.CONCATENATE, 0)
loopOutput1.set_input(1, inputT1)
#------------------------------------------------------------------------------- Network
network.mark_output(loopOutput0.get_output(0))
network.mark_output(loopOutput1.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_shape_input(1, t)  # 运行时绑定真实形状张量值
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput

bufferH = []
bufferH.append(data)
bufferH.append(np.ascontiguousarray(np.zeros([1], dtype=np.int32).reshape(-1)))  # 形状张量数据可用垃圾值
for i in range(nOutput):
    bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2(bufferD)
for i in range(nOutput):
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput):
    print("Input %d:" % i, bufferH[i].shape, "\n", bufferH[i])
for i in range(nOutput):
    print("Output %d:" % i, bufferH[nInput + i].shape, "\n", bufferH[nInput + i])

for buffer in bufferD:
    cudart.cudaFree(buffer)