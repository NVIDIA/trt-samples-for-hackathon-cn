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

import os
import numpy as np
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"
nGEMM = 10
sizeGEMM = 16
nInference = 10
np.random.seed(97)

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        inputList = []
        for i in range(nGEMM + 1):
            inputT = network.add_input("inputT" + str(i), trt.float32, [-1, 4, sizeGEMM, sizeGEMM])
            profile.set_shape(inputT.name, (1, 4, sizeGEMM, sizeGEMM), (4, 4, sizeGEMM, sizeGEMM), (sizeGEMM, 4, sizeGEMM, sizeGEMM))
            inputList.append(inputT)
        config.add_optimization_profile(profile)

        tempTensor = inputList[0]
        for i in range(1, nGEMM + 1):
            tempLayer = network.add_matrix_multiply(tempTensor, trt.MatrixOperation.NONE, inputList[i], trt.MatrixOperation.NONE)
            tempTensor = tempLayer.get_output(0)

        network.mark_output(tempLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    for i in range(nGEMM + 1):
        context.set_binding_shape(i, [4, 4, sizeGEMM, sizeGEMM])
    stream = cudart.cudaStreamCreate()[1]

    bufferSize = [trt.volume(context.get_binding_shape(i)) * np.array([0], dtype=trt.nptype(engine.get_binding_dtype(i))).nbytes for i in range(engine.num_bindings)]

    bufferH = []
    bufferD = []
    for i in range(nGEMM + 2):
        bufferH.append(cudart.cudaHostAlloc(bufferSize[i], cudart.cudaHostAllocWriteCombined)[1])
        bufferD.append(cudart.cudaMallocAsync(bufferSize[i], stream)[1])

    # 不用 CUDA Graph 来执行
    for i in range(nGEMM + 1):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bufferD, stream)
    cudart.cudaMemcpyAsync(bufferH[-1], bufferD[-1], bufferSize[-1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    for n in range(nInference):
        for i in range(nGEMM + 1):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2(bufferD, stream)
        cudart.cudaMemcpyAsync(bufferH[-1], bufferD[-1], bufferSize[-1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    # 捕获 CUDA Graph 并运行
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for i in range(nGEMM + 1):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i], bufferSize[i], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bufferD, stream)
    cudart.cudaMemcpyAsync(bufferH[-1], bufferD[-1], bufferSize[-1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    #cudart.cudaStreamSynchronize(stream)                       # 不用在 graph 内同步
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)

    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for n in range(nInference):
        cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for i in range(nGEMM + 2):
        cudart.cudaFree(bufferD[i])
    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cudart.cudaDeviceSynchronize()
    run()  # 创建 TensorRT 引擎并推理
    run()  # 读取 TensorRT 引擎并推理
