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
import os
import tensorrt as trt

dataFile = "./data.npz"
dataName = "data"
dataShape = [4, 4, 4, 4]
soFile = "./LoadNpzPlugin.so"
np.set_printoptions(precision=3, linewidth=100, suppress=True)
np.random.seed(31193)
cudart.cudaDeviceSynchronize()

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def createData():
    dataDict = {}
    dataDict[dataName] = np.ones(dataShape, dtype=np.float32)
    np.savez(dataFile, **dataDict)
    print("Succeeded saving data as .npz file!")
    return

def LoadNpzCPU(dummyInputTensor):
    return np.load(dataFile)[dataName]

def getLoadNpzPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "LoadNpzPlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    trtFile = "./model.plan"
    print("Test")
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
        config = builder.create_builder_config()

        #inputT0 = network.add_input("inputT0", trt.float32, [1])  # dummy input
        # Plugin Layer must have a input tensor, or we will get the error:
        # [TRT] [E] 2: [stdArchiveReader.h::readManyHelper::333] Error Code 2: Internal Error (Assertion prefix.count failed. Enums must always have at least one entry.)
        #pluginLayer = network.add_plugin_v2([inputT0], getLoadNpzPlugin())
        pluginLayer = network.add_plugin_v2([], getLoadNpzPlugin())
        network.mark_output(pluginLayer.get_output(0))
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
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.array([0], dtype=np.float32))
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

    outputCPU = LoadNpzCPU(bufferH[:nInput])

    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(outputCPU[i - nInput])

    check(bufferH[nInput:], outputCPU, True)

    for b in bufferD:
        cudart.cudaFree(b)
    print("Test finish!\n")

if __name__ == "__main__":
    os.system("rm -rf ./*.plan ./*.npz")

    createData()

    run()

    print("Test all finish!")
