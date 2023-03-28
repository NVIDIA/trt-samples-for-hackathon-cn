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

soFile = "./MultinomialDistributionPlugin.so"
np.random.seed(31193)

def printArrayInfomation(x, info="", n=10):
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

def getMultinomialDistributionPlugin(nCol, seed):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "MultinomialDistribution":
            parameterList = []
            parameterList.append(trt.PluginField("seed", np.int32(seed), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(nBatchSize, nCol, seed):
    testCase = "<nRow=%d,nCol=%s,seed=%d>" % (nBatchSize, nCol, seed)
    trtFile = "./model-nCol%d-seed-%d.plan" % (nCol, seed)
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

        inputT0 = network.add_input("inputT0", trt.float32, [-1, nCol])
        profile.set_shape(inputT0.name, [1, nCol], [32, nCol], [1024, nCol])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getMultinomialDistributionPlugin(nCol, seed))
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

    context = engine.create_execution_context()

    data = np.full([nBatchSize, nCol], 1, dtype=np.float32)  # uniform distribution
    #data = np.tile(np.arange(0,nCol,1,dtype=np.float32),[nBatchSize,1])  # non-uniform distribution
    context.set_binding_shape(0, [nBatchSize, nCol])
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    #for i in range(nInput):
    #    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    #for i in range(nInput, nInput + nOutput):
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(data.astype(np.float32).reshape(nBatchSize, nCol))
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
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput + i])
    """
    count, _ = np.histogram(bufferH[nInput], np.arange(nCol + 1))
    for i in range(nCol):
        print("[%3d]:%4d ---- %.3f %%" % (i, count[i], count[i] / nBatchSize * 100))

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run(1024, 4, 97)
    run(1024, 32, 97)
    run(1024, 128, 97)
    run(1024, 4, 89)
    run(1024, 32, 89)
    run(1024, 128, 89)

    print("Test all finish!")
