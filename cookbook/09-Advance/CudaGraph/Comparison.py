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
import os
import tensorrt as trt

trtFile = "./model.plan"
nGEMM = 10
sizeGEMM = 16
nInference = 10
np.random.seed(31193)

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
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    for i in range(nGEMM + 1):
        context.set_binding_shape(i, [4, 4, sizeGEMM, sizeGEMM])
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    #bufferSize = [trt.volume(context.get_binding_shape(i)) * np.array([0], dtype=trt.nptype(engine.get_binding_dtype(i))).nbytes for i in range(engine.num_bindings)]
    bufferH = []
    for i in range(nGEMM + 1):
        bufferH.append(np.random.rand(4 * 4 * sizeGEMM * sizeGEMM).astype(np.float32).reshape(4, 4, sizeGEMM, sizeGEMM))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # 不用 CUDA Graph 来执行
    _, stream = cudart.cudaStreamCreate()

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bufferD, stream)
    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    for n in range(nInference):
        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            context.execute_async_v2(bufferD, stream)
        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    # 捕获 CUDA Graph 并运行
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bufferD, stream)
    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    #cudart.cudaStreamSynchronize(stream)  # 不用在 graph 内同步
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)

    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for n in range(nInference):
        cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    cudart.cudaStreamDestroy(stream)
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cudart.cudaDeviceSynchronize()
    run()
    run()
