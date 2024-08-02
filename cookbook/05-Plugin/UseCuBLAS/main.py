#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

soFile = "./CuBLASGemmPlugin.so"
b, m, k, n = 5, 2, 3, 4
globalData = np.random.rand(b * m * k).astype(np.float32).reshape(b, m, k) * 2 - 1
globalWeight = np.random.rand(k * n).astype(np.float32).reshape(k, n) * 2 - 1
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

def CuBLASGemmCPU(inputH, weight):
    return [np.matmul(inputH[0], weight)]

def getCuBLASGemmPlugin(weight):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "CuBLASGemm":
            parameterList = []
            parameterList.append(trt.PluginField("weight", np.float32(weight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("k", np.int32(weight.shape[0]), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("n", np.int32(weight.shape[1]), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run():
    trtFile = "./model.plan"
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Fail loading engine")
            return
        print("Succeed loading engine")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, k])
        profile.set_shape(inputT0.name, [1, 1, k], [b, m, k], [b * 2, m * 2, k])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getCuBLASGemmPlugin(globalWeight))
        pluginLayer.get_output(0).name = "GEMM-Plugin-Output"

        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Fail building engine")
            return
        print("Succeed building engine")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [b, m, k])
    #for i in range(nIO):
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(globalData)
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

    outputCPU = CuBLASGemmCPU(bufferH[:nInput], globalWeight)
    """
    for i in range(nInput):
        printArrayInformation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInformation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInformation(outputCPU[i - nInput])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")

    run()  # build TensorRT engine and do inference
    run()  # load TensorRT engine and do inference

    print("Test all finish!")
