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

soFile = "./LayerNormPluginCUB.so"
epsilon = 1e-6
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

def layerNormCPU(bufferH, epsilon):
    _x, gamma, beta = bufferH
    nHiddenSize = bufferH[0].shape[2]
    _0 = np.mean(_x, 2)[:, :, np.newaxis]
    _1 = _x - _0
    _2 = _1 * _1
    _3 = np.mean(_2, 2)[:, :, np.newaxis]
    _4 = np.array(epsilon, dtype=np.float32)
    _5 = _4.reshape(1, 1, 1)
    _6 = _3 + _5
    _7 = np.sqrt(_6)
    _8 = 1 / _7  # 1/sqrt(...)
    _9 = gamma
    _10 = _9.reshape(1, 1, nHiddenSize)
    _11 = _8 * _10  # gamma/sqrt(...)
    _12 = _0 * _11  # bμ/sqrt(...)
    _13 = beta
    _14 = _13.reshape(1, 1, nHiddenSize)
    _15 = _14 - _12  # beta-bμ/sqrt(...)
    _16 = _x * _11  # bx/sqrt(...)
    _17 = _15 + _16  # gamma(x-μ)/sqrt(...)+beta
    _18 = _17.reshape(bufferH[0].shape[0], bufferH[0].shape[1], bufferH[0].shape[2])
    return [_18]

def getLayerNormPlugin(epsilon):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "LayerNorm" and c.plugin_version == "1":
            print("Find %s V%s" % (c.name, c.plugin_version))
            parameterList = []
            parameterList.append(trt.PluginField("epsilon", np.float32(epsilon), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, bFp16):
    testCase = "<shape=%s,dataType=%s>" % (shape, "FP16" if bFp16 else "FP32")
    trtFile = "./model-%d-%s.plan" % (shape[2], "FP16" if bFp16 else "FP32")
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if bFp16:
            config.set_flag(trt.BuilderFlag.FP16)

        inputT0 = network.add_input("inputT0", trt.float16 if bFp16 else trt.float32, [-1 for i in shape])
        profile.set_shape(inputT0.name, [1, 1, shape[2]], shape, shape)
        inputT1 = network.add_input("inputGamma", trt.float16 if bFp16 else trt.float32, [256])
        inputT2 = network.add_input("inputBeta", trt.float16 if bFp16 else trt.float32, [256])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0, inputT1, inputT2], getLayerNormPlugin(epsilon))
        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, shape)
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    #for i in range(nInput):
    #    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    #for i in range(nInput, nInput + nOutput):
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(np.random.rand(np.prod(shape)).astype(np.float16 if bFp16 else np.float32).reshape(shape) * 2 - 1)
    #bufferH.append(np.arange(np.prod(shape)).astype(np.float16 if bFp16 else np.float32).reshape(shape))
    bufferH.append(np.ones(shape[2]).astype(np.float16 if bFp16 else np.float32))
    bufferH.append(np.zeros(shape[2]).astype(np.float16 if bFp16 else np.float32))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = layerNormCPU(bufferH[:nInput], epsilon)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(outputCPU[i - nInput])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    os.system("rm -rf ./*.plan")
    run([16, 64, 32], False)
    os.system("rm -rf ./*.plan")
    run([16, 64, 32], True)

    os.system("rm -rf ./*.plan")
    run([16, 64, 256], False)
    os.system("rm -rf ./*.plan")
    run([16, 64, 256], True)

    os.system("rm -rf ./*.plan")
    run([16, 64, 1024], False)
    os.system("rm -rf ./*.plan")
    run([16, 64, 1024], True)

    os.system("rm -rf ./*.plan")
    run([16, 64, 1600], False)
    os.system("rm -rf ./*.plan")
    run([16, 64, 1600], True)

    print("Test all finish!")
