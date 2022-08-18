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

from cuda import cudart
import numpy as np

import tensorrt as trt

nB, nC, nH, nW = 8, 3, 224, 224
np.random.seed(97)
np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile0 = builder.create_optimization_profile()
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

inputT0 = network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
layer.get_output(0).name = "outputT0"
network.mark_output(layer.get_output(0))

profile0.set_shape(inputT0.name, (1, nC, nH, nW), (nB, nC, nH, nW), (nB * 2, nC, nH, nW))
profile1.set_shape(inputT0.name, (1, nC, nH, nW), (nB, nC, nH, nW), (nB * 2, nC, nH, nW))
config.add_optimization_profile(profile0)
config.add_optimization_profile(profile1)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
stream = 0  # 使用默认 CUDA 流

# 使用 Profile 0
print("Use Profile 0")
context.set_optimization_profile_async(0, stream)
cudart.cudaStreamSynchronize(stream)
#context.active_optimization_profile = 0  # deprecated since TRT8.4
context.set_binding_shape(0, [nB, nC, nH, nW])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # 获取 engine 绑定信息
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput + nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

data = np.random.rand(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW)
bufferH = []
bufferH.append(np.ascontiguousarray(data))
for i in range(nInput, nInput + nOutput):
    bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
bufferD = []
for i in range(nInput + nOutput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):  # 首先将 Host 数据拷贝到 Device 端
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)  # 运行推理计算

for i in range(nInput, nInput + nOutput):  # 将结果从 Device 端拷回 Host 端
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("check result:", np.all(bufferH[0] + bufferH[1] == 0))

# 使用 Profile 1
print("Use Profile 1")
context.set_optimization_profile_async(1, stream)
cudart.cudaStreamSynchronize(stream)
#context.active_optimization_profile = 1  # 与上面两行等价的选择 profile 的方法，不需要用 stream，但是将被废弃
context.set_binding_shape(2, [nB, nC, nH, nW])
print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

inputH1 = np.ascontiguousarray(data.reshape(-1))
outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
_, inputD1 = cudart.cudaMalloc(inputH1.nbytes)
_, outputD1 = cudart.cudaMalloc(outputH1.nbytes)

cudart.cudaMemcpy(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(0), int(0), int(inputD1), int(outputD1)])
cudart.cudaMemcpy(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("check result:", np.all(outputH0 == -inputH0.reshape(nB, nC, nH, nW)))

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
cudart.cudaFree(inputD1)
cudart.cudaFree(outputD1)
