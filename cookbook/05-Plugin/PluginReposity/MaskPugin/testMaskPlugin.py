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

import os
import ctypes
import numpy as np
#from time import time_ns
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath = "./MaskPlugin.so"

np.random.seed(31193)

npToTRT = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def maskCPU(bufferH):
    input0, input1 = bufferH
    bs, sl, _ = input0.shape
    negValue = [-3.0e38, -6.0e4][int(input0.dtype == np.float16)]

    output0 = np.zeros([bs, 4, sl, sl], dtype=input0.dtype) + 0
    output1 = np.zeros([bs, 4, sl, sl], dtype=input0.dtype) + negValue
    output2 = np.zeros([bs, sl, 320], dtype=input0.dtype) + 0

    for i in range(bs):
        validWidth = input1[i]
        output0[i, :, :validWidth, :validWidth] = 1
        output1[i, :, :validWidth, :validWidth] = 0
        output2[i, :validWidth, :] = 1

    return output0, output1, output2

def getMaskPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "MaskPlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def buildEngine(logger, datatype):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << 0)
    config = builder.create_builder_config()
    config.flags = [0, 1 << int(trt.BuilderFlag.FP16)][int(datatype == np.float16)]

    inputT0 = network.add_input("inputT0", npToTRT[datatype], [-1, -1, 560])
    inputT1 = network.add_input("inputT1", npToTRT[np.int32], [-1])

    profile = builder.create_optimization_profile()
    profile.set_shape(inputT0.name, [1, 1, 560], [2, 4, 560], [4, 8, 560])
    profile.set_shape(inputT1.name, [1], [2], [4])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2([inputT0, inputT1], getMaskPlugin())

    pluginLayer.get_output(0).dtype = npToTRT[datatype]
    pluginLayer.get_output(1).dtype = npToTRT[datatype]
    pluginLayer.get_output(2).dtype = npToTRT[datatype]

    network.mark_output(pluginLayer.get_output(0))
    network.mark_output(pluginLayer.get_output(1))
    network.mark_output(pluginLayer.get_output(2))
    return builder.build_engine(network, config)

def run(datatype, nBS, nSL):
    testCase = "test<fp%s,bs=%d,sl=%d>" % (["32", "16"][int(datatype == np.float16)], nBS, nSL)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = "./model-fp" + ["32", "16"][int(datatype == np.float16)] + ".plan"
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        engine = buildEngine(logger, datatype)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engine.serialize())

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBS, nSL, 560])
    context.set_binding_shape(1, [nBS])
    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    stream = cuda.Stream()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

    bufferH = []
    bufferH.append(np.random.rand(nBS * nSL * 560).reshape(nBS, nSL, 560).astype(datatype))
    bufferH.append(1 + np.arange(nBS).reshape(nBS).astype(np.int32))

    bufferH.append(np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2))))
    bufferH.append(np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3))))
    bufferH.append(np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cuda.mem_alloc(bufferH[i].nbytes))

    for i in range(nInput):
        cuda.memcpy_htod_async(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)), stream)

    context.execute_async_v2(bufferD, stream.handle)

    for i in range(nOutput):
        cuda.memcpy_dtoh_async(bufferH[nInput + i], bufferD[nInput + i], stream)

    stream.synchronize()

    for i in range(nInput):
        temp = bufferH[i]
        print( 'input%d: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
            i,str(temp.shape),np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))) ))
        print("\t", temp.reshape(-1)[:10])
    for i in range(nOutput):
        temp = bufferH[nInput + i]
        print( 'output%d: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
            i,str(temp.shape),np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))) ))
        print("\t", temp.reshape(-1)[:10])

    cpu = maskCPU(bufferH[:2])
    for i in range(nOutput):
        temp = bufferH[nInput + i] - cpu[i]
        print( 'diff%d: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
            i,str(temp.shape),np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))) ))
        print("\t", temp.reshape(-1)[:10])

    print("Test", testCase, "finish!")

if __name__ == "__main__":
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    #cuda.Device(0).make_context()

    #testEncoderCPU()

    run(np.float32, 4, 8)
    run(np.float16, 4, 8)

    #cuda.Context.pop()
    #print("test all finish!")
