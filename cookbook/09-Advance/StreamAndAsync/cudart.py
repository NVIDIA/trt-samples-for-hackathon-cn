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
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"
nC, nH, nW = 3, 4, 5

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        profile.set_shape(inputTensor.name, [1, 1, 1], [nC, nH, nW], [nC * 2, nH * 2, nW * 2])
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nC, nH, nW])
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    npData = []
    bufferSize = []
    bufferH = []
    bufferD = []

    for i in range(nInput):  # 输入 numpy 数组和返回 numpy 数组
        bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
        npData.append(np.arange(nC * nH * nW, dtype=np.float32).reshape(nC, nH, nW))
    for i in range(nInput, nInput + nOutput):
        bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
        npData.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))

    for i in range(nInput + nOutput):  # 申请 Host 端页锁定内存和 Device 端显存
        bufferH.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
        bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])

    for i in range(nInput):  # numpy 数组 -> 页锁定内存
        cudart.cudaMemcpyAsync(bufferH[i], npData[i].ctypes.data, bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferH, stream)  # 直接使用页锁定内存

    for i in range(nInput, nInput + nOutput):  # 页锁定内存 -> 返回 numpy 数组
        cudart.cudaMemcpyAsync(npData[i].ctypes.data, bufferH[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(npData[i].reshape(context.get_binding_shape(i)))

    for b in bufferH:
        cudart.cudaFreeHost(b)
    for b in bufferD:
        cudart.cudaFreeAsync(b, stream)
    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cudart.cudaDeviceSynchronize()
    run()
    run()
