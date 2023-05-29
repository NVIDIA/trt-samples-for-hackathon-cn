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

import ctypes
from cuda import cudart
import numpy as np
import nvtx
import os
import tensorrt as trt

trtFile = "./model.plan"
nB, nC, nH, nW = 1, 3, 256, 256
nTest = 30

def printArrayInfomation(x, info="", n=5):
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

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
        profile.set_shape(inputTensor.name, [nB, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC * 2, nH * 2, nW * 2])
        config.add_optimization_profile(profile)

        identityLayer = network.add_unary(inputTensor, trt.UnaryOperation.NEG)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    return context

def run(context, bUsePinnedMemory):
    engine = context.engine
    _, stream = cudart.cudaStreamCreate()
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    if bUsePinnedMemory:  # pin-memory needed for Async API
        bufferSize = []
        bufferH = []
        bufferD = []

        for i in range(nInput):
            bufferSize.append(trt.volume(context.get_tensor_shape(lTensorName[i])) * engine.get_tensor_dtype(lTensorName[i]).itemsize)
            bufferD.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
            pBufferCtype = ctypes.cast(bufferD[i], ctypes.POINTER(ctypes.c_float * trt.volume(context.get_tensor_shape(lTensorName[i]))))
            bufferH.append(np.ndarray(shape=context.get_tensor_shape(lTensorName[i]), buffer=pBufferCtype[0], dtype=np.float32))
            buffer = bufferH[-1].reshape(-1)
            for j in range(trt.volume(context.get_tensor_shape(lTensorName[i]))):
                buffer[j] = j
        for i in range(nInput, nIO):
            bufferSize.append(trt.volume(context.get_tensor_shape(lTensorName[i])) * engine.get_tensor_dtype(lTensorName[i]).itemsize)
            bufferD.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
            pBufferCtype = ctypes.cast(bufferD[-1], ctypes.POINTER(ctypes.c_float * trt.volume(context.get_tensor_shape(lTensorName[i]))))
            bufferH.append(np.ndarray(shape=context.get_tensor_shape(lTensorName[i]), buffer=pBufferCtype[0], dtype=np.float32))

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))  # use pin-memory directly

        # warm up
        context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)

        # test
        with nvtx.annotate("Pagelock", color="green"):
            for k in range(nTest):
                context.execute_async_v3(stream)
            cudart.cudaStreamSynchronize(stream)

        for i in range(nIO):
            printArrayInfomation(bufferH[i])

        for b in bufferH:
            cudart.cudaFreeHost(b)
        for b in bufferD:
            cudart.cudaFreeAsync(b, stream)
        cudart.cudaStreamDestroy(stream)

    else:  # do not use pin-memory
        bufferSize = []
        bufferH = []
        bufferD = []

        for i in range(nInput):
            bufferSize.append(trt.volume(context.get_tensor_shape(lTensorName[i])) * engine.get_tensor_dtype(lTensorName[i]).itemsize)
            bufferH.append(np.arange(nB * nC * nH * nW, dtype=np.float32).reshape(nC, nH, nW))
            bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])
        for i in range(nInput, nIO):
            bufferSize.append(trt.volume(context.get_tensor_shape(lTensorName[i])) * engine.get_tensor_dtype(lTensorName[i]).itemsize)
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
            bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])

        # warm up --------------------------------------------------------------
        for i in range(nInput):  # numpy array -> GPU memory
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        context.execute_async_v2(bufferD, stream)  # use GPU memory

        for i in range(nInput, nIO):  # GPU memory ->  numpy array
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        cudart.cudaStreamSynchronize(stream)

        # test -----------------------------------------------------------------
        with nvtx.annotate("Pageable", color="Red"):
            for k in range(nTest):
                for i in range(nInput):  # numpy array -> GPU memory
                    cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

                context.execute_async_v2(bufferD, stream)  # use GPU memory

                for i in range(nInput, nIO):  # GPU memory ->  numpy array
                    cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

            cudart.cudaStreamSynchronize(stream)

        for i in range(nIO):
            printArrayInfomation(bufferH[i])

        for b in bufferD:
            cudart.cudaFreeAsync(b, stream)
        cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cudart.cudaDeviceSynchronize()
    context = build()  # build engine and prepare context
    run(context, False)  # use pageable memory
    run(context, True)  # use pagelocked memory
