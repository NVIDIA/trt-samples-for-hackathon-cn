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

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        profile.set_shape(inputTensor.name, (1, 1, 1), (3, 4, 5), (6, 8, 10))
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)
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
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, [3, 4, 5])
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # 运行推理和使用 CUDA Graph 要用的流
    _, stream = cudart.cudaStreamCreate()

    # 捕获 CUDA Graph 之前需要先运行一次推理
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(bufferH[i].reshape(context.get_binding_shape(i)))

    # 首次捕获 CUDA Graph 并运行推理
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

    # 输入尺寸改变后，也需要首先运行一次推理，然后重新捕获 CUDA Graph，最后再运行推理
    context.set_binding_shape(0, [2, 3, 4])

    # 这里偷懒，因为本次推理绑定的输入输出数据形状不大于上一次推理，所以这里不再重新准备 bufferD
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(bufferH[i].reshape(context.get_binding_shape(i)))

    # 再次捕获 CUDA Graph 并运行推理
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bufferD, stream)
    for i in range(nInput, nInput + nOutput):  # 将结果从 Device 端拷回 Host 端
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    #cudart.cudaStreamSynchronize(stream)  # 不用在 graph 内同步
    _, graph = cudart.cudaStreamEndCapture(stream)
    _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)

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
