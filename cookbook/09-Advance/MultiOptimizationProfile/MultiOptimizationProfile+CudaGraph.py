#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import ctypes
import numpy as np
from cuda import cudart
import tensorrt as trt

np.random.seed(97)
nB, nC, nH, nW = 1, 3, 5, 5
data = np.random.rand(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW)

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
network.mark_output(layer.get_output(0))

profile.set_shape(inputT0.name, (1, 1, 1, 1), (nB, nC, nH, nW), (nB * 2, nC * 2, nH * 2, nW * 2))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
_, stream0 = cudart.cudaStreamCreate()
_, stream1 = cudart.cudaStreamCreate()
context = engine.create_execution_context_without_device_memory()
context.set_binding_shape(0, [nB * 2, nC * 2, nH * 2, nW * 2])
print("Context0 binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))

for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

inputH = np.empty(context.get_binding_shape(0), dtype=trt.nptype(engine.get_binding_dtype(0)))
outputH = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD = cudart.cudaMallocAsync(inputH.nbytes, stream0)
_, outputD = cudart.cudaMallocAsync(outputH.nbytes, stream0)
for i in range(np.prod(inputH.shape)):
    inputH.reshape(-1)[i] = 0

# do something on context
context.set_binding_shape(0, [nB, nC, nH, nW])

# first execute
cudart.cudaMemcpyAsync(inputD, inputH.ctypes.data, inputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
context.execute_async_v2([int(inputD), int(outputD)], stream0)
cudart.cudaMemcpyAsync(outputH.ctypes.data, outputD, outputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)
cudart.cudaStreamSynchronize(stream0)

# capture CUDA Graph and run
cudart.cudaStreamBeginCapture(stream0, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
cudart.cudaMemcpyAsync(inputD, inputH.ctypes.data, inputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
context.execute_async_v2([int(inputD), int(outputD)], stream0)
cudart.cudaMemcpyAsync(outputH.ctypes.data, outputD, outputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)
#cudart.cudaStreamSynchronize(stream)  # 不用在 graph 内同步
_, graph0 = cudart.cudaStreamEndCapture(stream0)
_, graphExe0, _ = cudart.cudaGraphInstantiate(graph0, b"", 0)

cudart.cudaGraphLaunch(graphExe0, stream0)
cudart.cudaStreamSynchronize(stream0)

# do something on context
context.set_binding_shape(0, [nB * 2, nC * 2, nH * 2, nW * 2])

# first execute
cudart.cudaMemcpyAsync(inputD, inputH.ctypes.data, inputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)
context.execute_async_v2([int(inputD), int(outputD)], stream1)
cudart.cudaMemcpyAsync(outputH.ctypes.data, outputD, outputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)
cudart.cudaStreamSynchronize(stream1)

# capture CUDA Graph and run
cudart.cudaStreamBeginCapture(stream1, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
cudart.cudaMemcpyAsync(inputD, inputH.ctypes.data, inputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)
context.execute_async_v2([int(inputD), int(outputD)], stream1)
cudart.cudaMemcpyAsync(outputH.ctypes.data, outputD, outputH.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)
#cudart.cudaStreamSynchronize(stream)  # 不用在 graph 内同步
_, graph1 = cudart.cudaStreamEndCapture(stream1)
_, graphExe1, _ = cudart.cudaGraphInstantiate(graph1, b"", 0)

cudart.cudaGraphLaunch(graphExe1, stream1)
cudart.cudaStreamSynchronize(stream1)

# test part
for i in range(np.prod(inputH.shape)):
    inputH.reshape(-1)[i] = i

cudart.cudaGraphLaunch(graphExe0, stream0)
cudart.cudaStreamSynchronize(stream0)
print("graph0, test0:", outputH.shape)
print(outputH.reshape(-1)[:np.prod([nB, nC, nH, nW])].reshape(nB, nC, nH, nW))

for i in range(np.prod(inputH.shape)):
    inputH.reshape(-1)[i] = i * 2

cudart.cudaGraphLaunch(graphExe1, stream1)
cudart.cudaStreamSynchronize(stream1)
print("graph1, test1:", outputH.shape)
print(outputH)

for i in range(np.prod(inputH.shape)):
    inputH.reshape(-1)[i] = i * 3

cudart.cudaGraphLaunch(graphExe0, stream0)
cudart.cudaStreamSynchronize(stream0)
print("graph0, test2:", outputH.shape)
print(outputH.reshape(-1)[:np.prod([nB, nC, nH, nW])].reshape(nB, nC, nH, nW))

for i in range(np.prod(inputH.shape)):
    inputH.reshape(-1)[i] = i * 4

cudart.cudaGraphLaunch(graphExe1, stream1)
cudart.cudaStreamSynchronize(stream1)
print("graph1, test3:", outputH.shape)
print(outputH)

# later part
cudart.cudaStreamDestroy(stream0)
cudart.cudaStreamDestroy(stream1)
cudart.cudaFree(inputD)
cudart.cudaFree(outputD)