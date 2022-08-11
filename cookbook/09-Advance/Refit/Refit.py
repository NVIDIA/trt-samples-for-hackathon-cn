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

nB, nC, nH, nW = 1, 1, 6, 9  # 输入张量 NCHW
nCOut, nKernelHeight, nKernelWidth = 1, 3, 3
data = np.tile(np.arange(1, 1 + nKernelHeight * nKernelWidth, dtype=np.float32).reshape(nKernelHeight, nKernelWidth), (nC, nH // nKernelHeight, nW // nKernelWidth)).reshape(nC, nH, nW)  # 输入张量
weight = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(nCOut, nKernelHeight, nKernelWidth)  # 卷积窗口
bias = np.zeros(nCOut, dtype=np.float32)  # 卷积偏置
trtFile = "./model.plan"

def run(nRunTime):
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
        config = builder.create_builder_config()
        config.flags = 1 << int(trt.BuilderFlag.REFIT)

        inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
        fakeWeight = np.zeros([nCOut, nC, nKernelWidth, nKernelWidth], dtype=np.float32)
        fakeBias = np.zeros([nCOut], dtype=np.float32)
        convolutionLayer = network.add_convolution_nd(inputT0, nCOut, (nKernelHeight, nKernelWidth), fakeWeight, fakeBias)
        #convolutionLayer.name = "conv"
        network.set_weights_name(convolutionLayer.kernel, "conv-w")
        network.set_weights_name(convolutionLayer.bias, "conv-b")

        network.mark_output(convolutionLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    if nRunTime == 0:
        print("Do not refit!")
    else:
        print("Refit!")
        refitter = trt.Refitter(engine, logger)
        refitter.set_named_weights("conv-w", weight)
        refitter.set_named_weights("conv-b", bias)

        [missingLayer, weightRole] = refitter.get_missing()
        for layer, role in zip(missingLayer, weightRole):
            print("[", layer, "-", role, "]")

        if refitter.refit_cuda_engine() == False:
            print("Failed Refitting engine!")
            return

    context = engine.create_execution_context()
    _, stream = cudart.cudaStreamCreate()
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    print("data:", data.shape)
    print(data)
    print("outputH0:", outputH0.shape)
    print(outputH0)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=8, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    run(0)
    run(1)
