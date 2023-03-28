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

from cuda import cudart
import numpy as np
import os
import tensorrt as trt

nB, nC, nH, nW = 1, 1, 6, 9
nCOut, nKernelHeight, nKernelWidth = 1, 3, 3
data = np.tile(np.arange(1, 1 + nKernelHeight * nKernelWidth, dtype=np.float32).reshape(nKernelHeight, nKernelWidth), (nC, nH // nKernelHeight, nW // nKernelWidth)).reshape(nC, nH, nW)
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
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.REFIT)

        inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
        profile.set_shape(inputT0.name, [1, nC, nH, nW], [2, nC, nH, nW], [4, nC, nH, nW])
        config.add_optimization_profile(profile)

        fakeWeight = np.zeros([nCOut, nC, nKernelWidth, nKernelWidth], dtype=np.float32)
        fakeBias = np.zeros([nCOut], dtype=np.float32)
        convolutionLayer = network.add_convolution_nd(inputT0, nCOut, (nKernelHeight, nKernelWidth), fakeWeight, fakeBias)
        convolutionLayer.name = "conv"

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
        print("Before refit!")
    else:
        print("After refit!")
        refitter = trt.Refitter(engine, logger)
        refitter.set_weights("conv", trt.WeightsRole.KERNEL, weight)
        refitter.set_weights("conv", trt.WeightsRole.BIAS, bias)

        [missingLayer, weightRole] = refitter.get_missing()
        for layer, role in zip(missingLayer, weightRole):
            print("[", layer, "-", role, "]")

        if refitter.refit_cuda_engine() == False:
            print("Failed Refitting engine!")
            return

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nB, nC, nH, nW])
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # 获取 engine 绑定信息
    nOutput = engine.num_bindings - nInput

    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("Input :\n", bufferH[0])
    print("Output:\n", bufferH[-1])

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=8, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()
    run(0)
    run(1)
