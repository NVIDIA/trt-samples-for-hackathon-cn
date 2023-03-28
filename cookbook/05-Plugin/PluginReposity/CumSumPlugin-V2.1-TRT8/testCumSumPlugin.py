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

soFile = "./CumSumPlugin.so"
dataTypeNpToTrt = {np.float32: trt.float32, np.float16: trt.float16, np.int32: trt.int32}

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-2):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def cumSumCPU(inputH, axis):
    return [np.cumsum(inputH[0], axis)]

def getCumSumPlugin(axis):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "CumSum":
            parameterList = []
            parameterList.append(trt.PluginField("axis", np.int32(axis), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, dataType, axis):
    if dataType == np.float32:
        dataTypeStr = "FP32"
    elif dataType == np.float16:
        dataTypeStr = "FP16"
    elif dataType == np.int32:
        dataTypeStr = "INT32"
    else:
        dataTypeStr = "Other"
    testCase = "<shape=%s,dataType=%s,axis=%d>" % (shape, dataTypeStr, axis)
    trtFile = "./model-%s-%s-%d.plan" % (shape, dataTypeStr, axis)
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
        if dataType == np.float16:
            config.set_flag(trt.BuilderFlag.FP16)

        inputT0 = network.add_input("inputT0", dataTypeNpToTrt[dataType], [-1 for i in shape])
        profile.set_shape(inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape[:-1]] + [256])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getCumSumPlugin(axis))
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
    if dataType == np.int32:
        bufferH.append(np.random.randint(-10, 10, shape).astype(np.int32).reshape(shape))
        #bufferH.append(np.arange(np.prod(shape)).astype(np.int32).reshape(shape))
    else:
        bufferH.append(np.random.rand(np.prod(shape)).astype(dataType).reshape(shape) * 2 - 1)
        #bufferH.append(np.arange(np.prod(shape)).astype(dataType).reshape(shape))
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

    outputCPU = cumSumCPU(bufferH[:nInput], axis)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
        print(bufferH[i])
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput+i])
    for i in range(nOutput):
        printArrayInfomation(outputCPU[i])
        print(bufferH[nInput+i])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    # w 维
    run([16], np.float32, 0)
    run([16], np.float16, 0)
    run([16], np.int32, 0)
    run([2, 16], np.float32, 1)
    run([2, 16], np.float16, 1)
    run([2, 16], np.int32, 1)
    run([2, 3, 16], np.float32, 2)
    run([2, 3, 16], np.float16, 2)
    run([2, 3, 16], np.int32, 2)
    run([2, 3, 4, 16], np.float32, 3)
    run([2, 3, 4, 16], np.float16, 3)
    run([2, 3, 4, 16], np.int32, 3)
    run([256], np.float32, 0)
    run([256], np.float16, 0)
    run([256], np.int32, 0)
    run([2, 256], np.float32, 1)
    run([2, 256], np.float16, 1)
    run([2, 256], np.int32, 1)
    run([2, 3, 256], np.float32, 2)
    run([2, 3, 256], np.float16, 2)  # 数据范围不足，产生 inf
    run([2, 3, 256], np.int32, 2)
    run([2, 3, 4, 256], np.float32, 3)
    run([2, 3, 4, 256], np.float16, 3)
    run([2, 3, 4, 256], np.int32, 3)

    # h 维
    run([2, 16], np.float32, 0)
    run([2, 16], np.float16, 0)
    run([2, 16], np.int32, 0)
    run([2, 3, 16], np.float32, 1)
    run([2, 3, 16], np.float16, 1)
    run([2, 3, 16], np.int32, 1)
    run([2, 3, 4, 16], np.float32, 2)
    run([2, 3, 4, 16], np.float16, 2)
    run([2, 3, 4, 16], np.int32, 2)

    run([2, 256], np.float32, 0)
    run([2, 256], np.float16, 0)
    run([2, 256], np.int32, 0)
    run([2, 3, 256], np.float32, 1)
    run([2, 3, 256], np.float16, 1)
    run([2, 3, 256], np.int32, 1)
    run([2, 3, 4, 256], np.float32, 2)
    run([2, 3, 4, 256], np.float16, 2)
    run([2, 3, 4, 256], np.int32, 2)

    # c 维
    run([2, 3, 16], np.float32, 0)
    run([2, 3, 16], np.float16, 0)
    run([2, 3, 16], np.int32, 0)
    run([2, 3, 4, 16], np.float32, 1)
    run([2, 3, 4, 16], np.float16, 1)
    run([2, 3, 4, 16], np.int32, 1)

    run([2, 3, 256], np.float32, 0)
    run([2, 3, 256], np.float16, 0)
    run([2, 3, 256], np.int32, 0)
    run([2, 3, 4, 256], np.float32, 1)
    run([2, 3, 4, 256], np.float16, 1)
    run([2, 3, 4, 256], np.int32, 1)

    # n 维
    run([2, 3, 4, 16], np.float32, 0)
    run([2, 3, 4, 16], np.float16, 0)
    run([2, 3, 4, 16], np.int32, 0)

    run([2, 3, 4, 256], np.float32, 0)
    run([2, 3, 4, 256], np.float16, 0)
    run([2, 3, 4, 256], np.int32, 0)

    print("Test all finish!")
