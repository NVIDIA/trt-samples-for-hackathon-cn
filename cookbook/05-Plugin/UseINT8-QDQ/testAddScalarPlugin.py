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
from calibrator import MyCalibrator

soFile = "./AddScalarPlugin.so"
cacheFile = "./int8.cache"
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

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, scalar):
    testCase = "<shape=%s,scalar=%f>" % (shape, scalar)
    trtFile = "./model-Dim%s.plan" % str(len(shape))
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
        config.set_flag(trt.BuilderFlag.INT8)
        #config.int8_calibrator = MyCalibrator(1, shape, cacheFile)

        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
        profile.set_shape(inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape])
        config.add_optimization_profile(profile)
        #inputT0.dynamic_range = [-100,100]  # set dynamic range if calibrator is not used

        q0Value = 100 / 128
        q0Tensor = network.add_constant([], np.array([q0Value], dtype=np.float32)).get_output(0)
        quantizeLayer = network.add_quantize(inputT0, q0Tensor)
        quantizeLayer.axis = 0

        pluginLayer = network.add_plugin_v2([quantizeLayer.get_output(0)], getAddScalarPlugin(scalar))
        pluginLayer.precision = trt.int8
        pluginLayer.set_output_type(0, trt.int8)
        pluginLayer.get_output(0).dtype = trt.int8
        #pluginLayer.get_output(0).dynamic_range = [-120,120]

        q1Value = 100 / 128
        q1Tensor = network.add_constant([], np.array([q1Value], dtype=np.float32)).get_output(0)
        dequantizeLayer = network.add_dequantize(pluginLayer.get_output(0), q1Tensor)
        dequantizeLayer.axis = 0

        network.mark_output(dequantizeLayer.get_output(0))
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
    context.set_input_shape(lTensorName[0], shape)
    #for i in range(nIO):
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.ascontiguousarray(np.arange(np.prod(shape), dtype=np.float32).reshape(shape)))
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

    outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(outputCPU[i - nInput])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for b in bufferD:
        cudart.cudaFree(b)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan ./*.cache")
    run([32], 0.1)

    os.system("rm -rf ./*.plan ./*.cache")  # cache files can not be shared among engines because input data ranges are different
    run([32, 32], 0.1)

    os.system("rm -rf ./*.plan ./*.cache")
    run([16, 16, 16], 0.1)  # CHW4 format needs input tensor with at least 4 Dimensions

    os.system("rm -rf ./*.plan ./*.cache")
    run([8, 8, 8, 8], 0.1)

    print("Test all finish!")
