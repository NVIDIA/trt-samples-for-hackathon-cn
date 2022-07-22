#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

soFilePath = "./LayerNormPlugin.so"
nBS = 4
nSL = 64
nEmbedding = 256
epsilon = 6e-6
npDataType = np.float32
np.random.seed(97)

def printArrayInfo(x, info="", n=5):
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
    print("check:", res, diff0, diff1)

def layerNormCPU(bufferH, epsilon):
    _x, = bufferH
    _0 = np.mean(_x, 2)[:, :, np.newaxis]
    _1 = _x - _0
    _2 = _1 * _1
    _3 = np.mean(_2, 2)[:, :, np.newaxis]
    _4 = np.array(epsilon, dtype=np.float32)
    _5 = _4.reshape(1, 1, 1)
    _6 = _3 + _5
    _7 = np.sqrt(_6)
    _8 = 1 / _7  # 1/sqrt(...)
    _9 = _1 * _8
    return [_9]

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm':
            p0 = trt.PluginField('epsilon', np.float32(epsilon), trt.PluginFieldType.FLOAT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def run():
    testCase = "%d-%d-%d-fp%s" % (nBS, nSL, nEmbedding, '16' if int(npDataType == np.float16) else '32')
    print("Test <%s>" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = "./model-" + testCase + ".plan"
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
        network = builder.create_network(1 << 0)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
        config.flags = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0

        inputTensorList = []
        trtDataType = trt.float16 if int(npDataType == np.float16) else trt.float32
        inputTensorList.append(network.add_input('inputT', trtDataType, [-1, -1, -1]))

        profile = builder.create_optimization_profile()
        profile.set_shape('inputT', [1, 1, nEmbedding], [nBS, nSL, nEmbedding], [nBS * 2, nSL * 2, nEmbedding])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
        pluginLayer.get_output(0).dtype = trtDataType

        network.mark_output(pluginLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBS, nSL, nEmbedding])

    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

    bufferH = []
    bufferH.append(np.random.rand(nBS, nSL, nEmbedding).astype(np.float32).reshape(nBS, nSL, nEmbedding) * 2 - 1)
    bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    resCPU = layerNormCPU(bufferH, epsilon)[-1]
    #printArrayInfo(resCPU)
    #printArrayInfo(bufferH[-1])
    check(bufferH[-1], resCPU, True)

    for b in bufferD:
        cudart.cudaFree(b)

    print("Test <%s> finish!" % testCase)

if __name__ == "__main__":
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    run()
