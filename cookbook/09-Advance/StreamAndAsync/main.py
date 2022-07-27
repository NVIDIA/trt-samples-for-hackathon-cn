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

import ctypes
import os
import numpy as np
import nvtx
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"
nC, nH, nW = 3, 256, 256
nTest = 30

def printArrayInfo(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def build():
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

        #identityLayer = network.add_identity(inputTensor)
        identityLayer = network.add_unary(inputTensor, trt.UnaryOperation.NEG)
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
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    return context

def run(context, bUsePinnedMemory):
    engine = context.engine
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput

    if bUsePinnedMemory:  # Async 类的函数需要配合页锁定内存来使用，否则强制变回同步类的函数
        bufferSize = []
        bufferH = []
        bufferD = []

        for i in range(nInput):
            bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
            bufferD.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
            pBufferCtype = ctypes.cast(bufferD[i], ctypes.POINTER(ctypes.c_float * trt.volume(context.get_binding_shape(i))))
            bufferH.append(np.ndarray(shape=context.get_binding_shape(i), buffer=pBufferCtype[0], dtype=np.float32))
            for j in range(trt.volume(context.get_binding_shape(i))):
                bufferH[i].reshape(-1)[j] = j
        for i in range(nInput, nInput + nOutput):
            bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
            bufferD.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
            pBufferCtype = ctypes.cast(bufferD[-1], ctypes.POINTER(ctypes.c_float * trt.volume(context.get_binding_shape(i))))
            bufferH.append(np.ndarray(shape=context.get_binding_shape(i), buffer=pBufferCtype[0], dtype=np.float32))

        # warm up --------------------------------------------------------------
        context.execute_async_v2(bufferD, stream)  # 直接使用 Pinned memory
        cudart.cudaStreamSynchronize(stream)

        # test -----------------------------------------------------------------
        with nvtx.annotate("Pagelock", color="green"):
            for k in range(nTest):
                context.execute_async_v2(bufferD, stream)  # 直接使用 Pinned memory
            cudart.cudaStreamSynchronize(stream)

        for i in range(nInput + nOutput):
            printArrayInfo(bufferH[i])

        for b in bufferH:
            cudart.cudaFreeHost(b)
        for b in bufferD:
            cudart.cudaFreeAsync(b, stream)
        cudart.cudaStreamDestroy(stream)

    else:  # 测试一下不使用页锁定内存的情况
        bufferSize = []
        bufferH = []
        bufferD = []

        for i in range(nInput):
            bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
            bufferH.append(np.arange(nC * nH * nW, dtype=np.float32).reshape(nC, nH, nW))
            bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])
        for i in range(nInput, nInput + nOutput):
            bufferSize.append(trt.volume(context.get_binding_shape(i)) * engine.get_binding_dtype(i).itemsize)
            bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
            bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])

        # warm up --------------------------------------------------------------
        for i in range(nInput):  # numpy 数组 -> 显存
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        context.execute_async_v2(bufferD, stream)  # 使用显存

        for i in range(nInput, nInput + nOutput):  # 显存 -> 返回 numpy 数组
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        cudart.cudaStreamSynchronize(stream)

        # test -----------------------------------------------------------------
        with nvtx.annotate("Pageable", color="Red"):
            for k in range(nTest):
                cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

                context.execute_async_v2(bufferD, stream)  # 使用显存

                for i in range(nInput, nInput + nOutput):  # 显存 -> 返回 numpy 数组
                    cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

            cudart.cudaStreamSynchronize(stream)

        for i in range(nInput + nOutput):
            printArrayInfo(bufferH[i])

        for b in bufferD:
            cudart.cudaFreeAsync(b, stream)
        cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    #os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    context = build()  # 构建 engine 并筹备 context
    run(context, 0)  # 使用分页内存（Pageable memory）
    run(context, 1)  # 使用页锁定内存（Pagelocked memory）
