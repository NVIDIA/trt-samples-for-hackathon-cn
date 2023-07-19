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
import os

import numpy as np
import tensorrt as trt
from cuda import cudart

soFile = "./AddSubMulPlugin.so"
np.set_printoptions(precision=3, linewidth=200, suppress=True)
np.random.seed(31193)
cudart.cudaDeviceSynchronize()

def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        print('%s:%s' % (info, str(x.shape)))
        print()
        return
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def AddSubMulCPU(inputList):

    a = inputList[0]
    b = inputList[1]
    nBatch = a.shape[0]
    nLengthA = a.shape[1]
    nLengthB = b.shape[1]
    nLength = min(nLengthA, nLengthB)

    res0 = np.zeros([nBatch, nLengthA, nLengthB], dtype=np.float32)
    for i in range(nBatch):
        res0[i] = np.matmul(a[i].reshape(-1, 1), b[i].reshape(1, -1))

    res1 = a[:, np.newaxis, :nLength] + b[:, np.newaxis, :nLength]

    return [res0, res1]

def getAddSubMulPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddSubMul":
            parameterList = []
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shapeA, shapeB):
    testCase = "<shapeA=%s,shapeB=%s>" % (shapeA, shapeB)
    trtFile = "./model.plan"
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
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
        config.set_flag(trt.BuilderFlag.FP16)

        inputT0 = network.add_input("inputT0", trt.float32, [-1, -1])
        profile.set_shape(inputT0.name, [1, 1], [4, 256], [16, 1024])
        inputT1 = network.add_input("inputT1", trt.float32, [-1, -1])
        profile.set_shape(inputT1.name, [1, 1], [4, 256], [16, 1024])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0, inputT1], getAddSubMulPlugin())
        network.mark_output(pluginLayer.get_output(0))
        network.mark_output(pluginLayer.get_output(1))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shapeA)
    context.set_input_shape(lTensorName[1], shapeB)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.arange(np.prod(shapeA), dtype=np.float32).reshape(shapeA) / 10000)
    bufferH.append(np.arange(np.prod(shapeB), dtype=np.float32).reshape(shapeB) / 10000)
    #bufferH.append(np.random.rand(np.prod(shapeA)).astype(np.float32).reshape(shapeA))
    #bufferH.append(np.random.rand(np.prod(shapeB)).astype(np.float32).reshape(shapeB))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = AddSubMulCPU(bufferH[:nInput])
    """
    for i in range(nInput):
        printArrayInformation(bufferH[i], "Input")
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i], "GPU")
    for i in range(nInput, nIO):
        printArrayInformation(outputCPU[i - nInput], "CPU")
    """
    for i in range(nIO - nInput):
        check(bufferH[nInput:][i], outputCPU[i], True, checkEpsilon=1e-3)

    for b in bufferD:
        cudart.cudaFree(b)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")

    run([1, 8], [1, 8])  # small, equal
    run([1, 256], [1, 256])  # medium, equal
    run([1, 500], [1, 500])  # large, equal, not the times of 256

    run([2, 8], [2, 24])  # small, not equal
    run([3, 256], [3, 300])  # medium, not equal
    run([4, 500], [4, 1000])  # large, equal, not the times of 256

    print("Test all finish!")
