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

soFile = "./ZeroPlugin.so"
np.random.seed(31193)

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

def ZeroCPU(inputH):
    return [np.zeros(inputH[0], dtype=np.float32)]

def getZeroPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "Zero":
            parameterList = []
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape):
    testCase = "<shape=%s>" % (shape)
    trtFile = "./model-Shape%s.plan" % str(shape)
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

        inputT0 = network.add_input("inputT0", trt.int32, [2])
        profile.set_shape_input(inputT0.name, [1, 1], [3, 4], [6, 8])
        config.add_optimization_profile(profile)

        baseLayer = network.add_constant([1, 1, 1], trt.Weights(np.zeros([1, 1, 1], dtype=np.float32)))

        zeroLayer = network.add_constant([1], np.array([0], dtype=np.int32))

        pqzLayer = network.add_concatenation([inputT0, zeroLayer.get_output(0)])
        pqzLayer.axis = 0

        sliceLayer = network.add_slice(baseLayer.get_output(0), [0, 0, 0], [0, 0, 0], [0, 0, 0])
        sliceLayer.set_input(2, pqzLayer.get_output(0))

        pluginLayer = network.add_plugin_v2([sliceLayer.get_output(0)], getZeroPlugin())

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
    nOutput = nIO - nInput
    context = engine.create_execution_context()
    #context.set_binding_shape(0, shape)
    context.set_shape_input(0, shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.ones([2], dtype=np.int32))
    for i in range(nOutput):
        bufferH.append(np.ones(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = ZeroCPU([np.array(shape, dtype=np.int32)])

    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(outputCPU[i - nInput])

    check(bufferH[nInput:][0], outputCPU[0], True)

    for b in bufferD:
        cudart.cudaFree(b)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run([3, 4])
    run([6, 7])

    print("Test all finish!")
