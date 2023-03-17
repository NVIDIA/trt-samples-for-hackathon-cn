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
import sys
import ctypes
import numpy as np
from scipy.special import expit as sigmoid
import tensorrt as trt
#import cupy.cuda as CD
import pycuda.driver as cuda
import pycuda.autoinit

np.random.seed(31193)
npToTrt = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}
nBatchSize = 2
maxSL = 40
nDimInput = 128
nDimHidden = 128
epsilonFP32 = 1.0e-5
epsilonFP16 = 1.0e-2
soFile = "./GruPlugin.so"
globalWeightFC = np.linspace(-0.5, 0.5, nDimInput * nDimHidden * 3, dtype=np.float32).reshape(nDimInput, nDimHidden * 3)
globalWeightGRU = np.linspace(-0.5, 0.5, nDimHidden * nDimHidden * 3, dtype=np.float32).reshape(nDimHidden, nDimHidden * 3)
globalBias = np.zeros((nDimHidden, 3), dtype=np.float32)

def check(a, b, weak=False):
    if weak:
        epsilon = [epsilonFP16, epsilonFP32][int(a.dtype == np.float32)]
        return np.all(np.abs(a - b) < epsilon)
    else:
        return np.all(a == b)

def gruCPU(inputH0, inputH1):
    weightFC = np.split(globalWeightFC, 3, axis=1)
    weightGRU = np.split(globalWeightGRU, 3, axis=1)
    hAllState = np.zeros([nBatchSize, maxSL, nDimHidden], dtype=np.float32)
    hLastState = np.zeros((nBatchSize, nDimHidden)).astype(np.float32)
    for k in range(nBatchSize):
        h_t = np.zeros([1, nDimHidden], dtype=np.float32)
        inp = inputH0[k]
        for i in range(inputH1[k]):
            x_t = inputH0[k, i]
            u_t = sigmoid(np.dot(x_t, weightFC[0]) + np.dot(h_t, weightGRU[0]))
            r_t = sigmoid(np.dot(x_t, weightFC[1]) + np.dot(h_t, weightGRU[1]))
            g_t = np.tanh(np.dot(x_t, weightFC[2]) + np.dot((r_t * h_t), weightGRU[2]))
            h_t = ((np.ones([1, nDimHidden], dtype=np.float32) - u_t) * h_t + u_t * g_t)
            hAllState[k, i] = h_t
        hLastState[k] = hAllState[k, inputH1[k] - 1]
    return hAllState, hLastState

def cleanTrash(inputH0, inputH1):
    for i in range(inputH0.shape[0]):
        inputH0[i, inputH1[i]:, :] = 0
    return inputH0

def getGruPlugin(nDimInput: int, nDimHidden: int, weightX: np.array, weightH: np.array, bias: np.array):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "GruPlugin":
            p0 = trt.PluginField("nDimInput", np.array([nDimInput], dtype=np.int32), trt.PluginFieldType.INT32)
            p1 = trt.PluginField("nDimHidden", np.array([nDimHidden], dtype=np.int32), trt.PluginFieldType.INT32)
            p2 = trt.PluginField("WeightX", weightX, trt.PluginFieldType.FLOAT32)
            p3 = trt.PluginField("WeightH", weightH, trt.PluginFieldType.FLOAT32)
            p4 = trt.PluginField("Bias", bias, trt.PluginFieldType.FLOAT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0, p1, p2, p3, p4]))
    return None

def buildEngine(logger, dataType):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        config.flags = int(dataType == np.float16)

    inputT0 = network.add_input("data", npToTrt[dataType], shape=[nBatchSize, maxSL, nDimInput])
    profile.set_shape(inputT0.name, [nBatchSize, maxSL, nDimInput], [nBatchSize, maxSL, nDimInput], [nBatchSize, maxSL, nDimInput])
    inputT1 = network.add_input("sequenceLength", trt.int32, shape=[nBatchSize])
    profile.set_shape(inputT1.name, [nBatchSize], [nBatchSize], [nBatchSize])
    config.add_optimization_profile(profile)

    weightGRU = np.split(globalWeightGRU, 3, axis=1)
    weightGRU = np.concatenate([weightGRU[0], weightGRU[1], weightGRU[2]], axis=0)
    gruPlugin = getGruPlugin(nDimInput, nDimHidden, globalWeightFC, weightGRU, globalBias)
    gru = network.add_plugin_v2([inputT0, inputT1], gruPlugin)
    gru.name = "GRU"
    if dataType == np.float32:
        gru.precision = trt.float32
        gru.set_output_type(0, trt.float32)
        gru.set_output_type(1, trt.float32)
    elif dataType == np.float16:
        gru.precision = trt.float16
        gru.set_output_type(0, trt.float16)
        gru.set_output_type(1, trt.float16)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    else:
        print("datatype not support!")

    network.mark_output(gru.get_output(0))
    network.mark_output(gru.get_output(1))
    return builder.build_engine(network, config)

def run(time, dataType):
    print("test", dataType, "%d time" % time)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)

    trtFile = "./model-fp" + ["32", "16"][int(dataType == np.float16)] + ".plan"
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if engine == None:
                print("Failed loading engine!")
                return None
            print("Succeeded loading engine!")
    else:
        engine = buildEngine(logger, dataType)
        if engine == None:
            print("Failed building engine!")
            return None
        print("Succeeded building engine!")
        engineStr = engine.serialize()
        with open(trtFile, "wb") as f:
            f.write(engineStr)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBatchSize, maxSL, nDimInput])
    context.set_binding_shape(1, [nBatchSize])
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
    print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
    print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))
    stream = cuda.Stream()

    data0 = np.random.rand(nBatchSize, maxSL, nDimInput)
    data1 = np.random.randint(low=1, high=maxSL + 1, size=[nBatchSize])

    inputH0 = data0.astype(trt.nptype(engine.get_binding_dtype(0)))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = data1.astype(trt.nptype(engine.get_binding_dtype(1)))
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputH1 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputD1 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, np.ascontiguousarray(inputH0), stream)
    cuda.memcpy_htod_async(inputD1, np.ascontiguousarray(inputH1), stream)
    #CD.nvtx.RangePush("gru")
    context.execute_async_v2([int(inputD0), int(inputD1), int(outputD0), int(outputD1)], stream.handle)
    #CD.nvtx.RangePop()
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    print("InputH0->", inputH0.shape, engine.get_binding_dtype(0))
    #print(inputH0)
    print("InputH1->", inputH1.shape, engine.get_binding_dtype(1))
    #print(inputH1)
    print("OutputH0->", outputH0.shape, engine.get_binding_dtype(2))
    #print(cleanTrash(outputH0,inputH1))
    print("OutputH1->", outputH1.shape, engine.get_binding_dtype(3))
    #print(outputH1)

    outputH0CPU, outputH1CPU = gruCPU(inputH0, inputH1)
    print(check(cleanTrash(outputH0, inputH1), cleanTrash(outputH0CPU, inputH1), True))
    print(check(outputH1, outputH1CPU, True))
    print("test", dataType, "%d time finish" % time)

if __name__ == "__main__":
    os.system("rm -rf ./engine*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    #cuda.Device(0).make_context()

    run(0, np.float32)
    #CD.profiler.start()
    run(1, np.float32)
    #CD.profiler.stop()

    run(0, np.float16)
    #CD.profiler.start()
    run(1, np.float16)
    #CD.profiler.stop()

    #cuda.Context.pop()
    print("test finish!")
