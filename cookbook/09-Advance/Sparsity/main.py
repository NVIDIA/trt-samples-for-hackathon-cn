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
from time import time_ns

trtFile = "./model.plan"
nB, nC, nH, nW = 16, 1, 64, 64
nCBig, nK = 256, 5
nLoop = 10
nWarmUp = 10
nTest = 100
np.random.seed(31193)
kernelUp = (np.random.rand(nCBig, nC, nK, nK).astype(np.float32) * 2 - 1) / nCBig / nK / nK
biasUp = np.random.rand(nCBig).astype(np.float32) * 2 - 1
kernelDown = (np.random.rand(nC, nCBig, nK, nK).astype(np.float32) * 2 - 1) / nC / nK / nK
biasDown = np.random.rand(nC).astype(np.float32) * 2 - 1

kernelUp = kernelUp.reshape(-1)
for i in range(0, kernelUp.shape[0], 2):
    kernelUp[i] = 0
kernelUp = kernelUp.reshape(nCBig, nC, nK, nK)

kernelDown = kernelDown.reshape(-1)
for i in range(0, kernelDown.shape[0], 2):
    kernelDown[i] = 0
kernelDown = kernelDown.reshape(nC, nCBig, nK, nK)

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def run(bUseSparsity):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 22 << 30)
    if bUseSparsity:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    inputTensor = network.add_input("inputT0", trt.float32, [-1, 1, -1, -1])
    profile.set_shape(inputTensor.name, [nB, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH * 2, nW * 2])
    config.add_optimization_profile(profile)

    tensor = inputTensor
    for i in range(nLoop):
        convolutionUpLayer = network.add_convolution_nd(tensor, nCBig, [nK, nK], kernelUp, biasUp)
        convolutionUpLayer.padding_nd = [nK // 2, nK // 2]
        tensor = convolutionUpLayer.get_output(0)

        convolutionDownLayer = network.add_convolution_nd(tensor, nC, [nK, nK], kernelDown, biasDown)
        convolutionDownLayer.padding_nd = [nK // 2, nK // 2]
        tensor = convolutionDownLayer.get_output(0)

    network.mark_output(tensor)
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nB, nC, nH, nW])
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.arange(np.prod([nB, nC, nH, nW]), dtype=np.float32).reshape(nB, nC, nH, nW) / np.prod([nB, nC, nH, nW]) - 0.5
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

    for i in range(nWarmUp):
        context.execute_v2(bufferD)

    t0 = time_ns()
    for i in range(nTest):
        context.execute_v2(bufferD)
    t1 = time_ns()
    print("Time per inference: %f ms" % ((t1 - t0) / 1000000 / nTest))

    printArrayInfomation(bufferH[-1])

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run(False)  # 不使用 Sparsity
    run(True)  # 不使用 Sparsity
