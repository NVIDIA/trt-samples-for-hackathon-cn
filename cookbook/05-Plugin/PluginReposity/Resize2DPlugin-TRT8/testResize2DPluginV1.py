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
from scipy import interpolate
import tensorrt as trt

soFile = "./Resize2DPlugin.so"
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

def addResizeCPU(inputH, nMode, nScale, nH1, nW1):
    nB, nC, nH0, nW0 = inputH[0].shape
    if nScale > 0 and nH1 == 0 and nW1 == 0:
        nH1, nW1 = nH0 * nScale, nW0 * nScale
    res = np.zeros([nB, nC, nH1, nW1], dtype=np.float32)

    if nMode == 0:  # nearest interpolation
        indexH = ((np.arange(nH1) + 0.5) * nH0 / nH1).astype(np.int32)
        indexW = ((np.arange(nW1) + 0.5) * nW0 / nW1).astype(np.int32)

        for b in range(nB):
            for c in range(nC):
                for h in range(nH1):
                    for w in range(nW1):
                        res[b, c, h, w] = inputH[0][b, c, indexH[h], indexW[w]]

    elif nMode == 1:  #  bilinear interpolation
        h0 = (1 / 2 + np.arange(nH0)) / nH0  # Half_pixel, align_corner
        w0 = (1 / 2 + np.arange(nW0)) / nW0
        h1 = (1 / 2 + np.arange(nH1)) / nH1
        w1 = (1 / 2 + np.arange(nW1)) / nW1
        h1[0], w1[0] = h0[0], w0[0]
        h1[-1], w1[-1] = h0[-1], w0[-1]
        for b in range(nB):
            for c in range(nC):
                res[b, c] = interpolate.interp2d(w0, h0, inputH[0][b, c], kind="linear")(w1, h1)

    else:
        print("[addResizeCPU]Error interpolation mode!")
        res = inputH[0]

    return [res]

def getResizePlugin(nMode, nScale, nH1, nW1):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "Resize2D" and c.plugin_version == "1":
            parameterList = []
            parameterList.append(trt.PluginField("Mode", np.int32(nMode), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("Scale", np.int32(nScale), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("OutputHeight", np.int32(nH1), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("OutputWidth", np.int32(nW1), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, nMode, nScale, nH1, nW1):
    testCase = "<shape=%s,nMode=%d,nScale=%f,nH1=%d,nW1=%d>" % (shape, nMode, nScale, nH1, nW1)
    trtFile = "./model-%d-%f-%d-%d.plan" % (nMode, nScale, nH1, nW1)
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

        inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
        profile.set_shape(inputT0.name, [1 for i in shape], shape, shape)
        config.add_optimization_profile(profile)

        resizeLayer = network.add_plugin_v2([inputT0], getResizePlugin(nMode, nScale, nH1, nW1))
        network.mark_output(resizeLayer.get_output(0))
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
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.tile(np.arange(shape[-1]).astype(np.float32).reshape(1, 1, 1, shape[-1]), [shape[0], shape[1], shape[2], 1])

    bufferH = []
    bufferH.append(data)
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

    outputCPU = addResizeCPU(bufferH[:nInput], nMode, nScale, nH1, nW1)
    '''
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
        print(bufferH[i])
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput + i])
        print(bufferH[nInput + i])
    for i in range(nOutput):
        printArrayInfomation(outputCPU[i])
        print(outputCPU)
    '''
    check(bufferH[nInput:][0], outputCPU[0], True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    # nearest interpolation
    os.system("rm -rf ./*.plan")
    run([2, 8, 256, 256], 0, 2, 0, 0)
    os.system("rm -rf ./*.plan")
    run([2, 8, 256, 256], 0, 0, 512, 510)

    # bilinear interpolation
    os.system("rm -rf ./*.plan")
    run([2, 8, 256, 256], 1, 2, 0, 0)
    os.system("rm -rf ./*.plan")
    run([2, 8, 256, 256], 1, 0, 510, 510)

    print("Test all finish!")
