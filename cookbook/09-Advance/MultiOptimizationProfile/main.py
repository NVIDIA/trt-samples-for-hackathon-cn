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

nB, nC, nH, nW = 2, 3, 4, 5
nProfile = 2  # 要使用的 OptimizationProfile 数量
np.random.seed(31193)
np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profileList = [builder.create_optimization_profile() for index in range(nProfile)]
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])  # 使用两输入一输出的网络做范例
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1, -1])
for profile in profileList:
    profile.set_shape(inputT0.name, (nB, nC, nH, nW), (nB, nC, nH, nW), (nB * nProfile, nC * nProfile, nH * nProfile, nW * nProfile))  # 这里形状中的 nProfile 只是本范例用到的范围，实际使用时根据需求设定范围即可
    profile.set_shape(inputT1.name, (nB, nC, nH, nW), (nB, nC, nH, nW), (nB * nProfile, nC * nProfile, nH * nProfile, nW * nProfile))
    config.add_optimization_profile(profile)

layer = network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)
network.mark_output(layer.get_output(0))

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

stream = 0  # 使用默认 CUDA 流
context = engine.create_execution_context()
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
nInput = nInput // nProfile
nOutput = nOutput // nProfile

bufferH = []
for index in range(nProfile):
    context.set_optimization_profile_async(index, stream)
    bindingPad = (nInput + nOutput) * index  # 跳过前面 OptimizationProfile 占用的 Binding
    bindingShape = (np.array([nB, nC, nH, nW]) * (index + 1)).tolist()  # Binding0 使用 [nB, nC, nH, nW]，Binding1 使用 [nB*2, nC*2, nH*2, nW*2]，以此类推
    context.set_binding_shape(bindingPad + 0, bindingShape)
    context.set_binding_shape(bindingPad + 1, bindingShape)
    print("Context%d binding all? %s" % (index, "Yes" if context.all_binding_shapes_specified else "No"))
    for i in range(engine.num_bindings):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

    for i in range(nInput):
        #bufferH.append(np.random.rand(*bindingShape).astype(np.float32) * 2 - 1)
        bufferH.append(np.arange(np.prod(bindingShape)).astype(np.float32).reshape(bindingShape))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(bindingPad + nInput + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + nInput + i))))

bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for index in range(nProfile):
    bindingPad = (nInput + nOutput) * index
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

for index in range(nProfile):
    print("Use Profile %d" % index)
    context.set_optimization_profile_async(index, stream)  # 重设 Profile 后需要重新绑定输入张量形状
    bindingPad = (nInput + nOutput) * index
    bindingShape = (np.array([nB, nC, nH, nW]) * (index + 1)).tolist()
    context.set_binding_shape(bindingPad + 0, bindingShape)
    context.set_binding_shape(bindingPad + 1, bindingShape)
    bindingPad = (nInput + nOutput) * index
    bufferList = [int(0) for b in bufferD[:bindingPad]] + [int(b) for b in bufferD[bindingPad:(bindingPad + nInput + nOutput)]] + [int(0) for b in bufferD[(bindingPad + nInput + nOutput):]]
    # 分为三段，除了本 Context 对应的 Binding 位置上的 bufferD 以外，全部填充 int(0)
    # 其实也可以直接 bufferList = bufferD，只不过除了本 Context 对应的 Binding 位置上的 bufferD 以外全都用不到
    context.execute_async_v2(bufferList, stream)

for index in range(nProfile):
    bindingPad = (nInput + nOutput) * index
    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[bindingPad + nInput + i].ctypes.data, bufferD[bindingPad + nInput + i], bufferH[bindingPad + nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

for index in range(nProfile):
    cudart.cudaStreamSynchronize(stream)

for index in range(nProfile):
    bindingPad = (nInput + nOutput) * index
    print("check result of context %d: %s" % (index, np.all(bufferH[bindingPad + 2] == bufferH[bindingPad + 0] + bufferH[bindingPad + 1])))

for b in bufferD:
    cudart.cudaFree(b)
