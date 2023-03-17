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
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

npToTrt = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}
soFilePath = "./MaxPlugin.so"

def maxCPU(inputH0, inputH1):
    outputH0CPU = np.zeros_like(inputH0[:, 0, :], dtype=np.float32)
    for i in range(inputH0.shape[0]):
        maxLine = np.full(inputH0.shape[-1], -600000, dtype=np.float32)
        for j in range(inputH1[i]):
            maxLine = np.maximum(maxLine, inputH0[i, j])
        outputH0CPU[i, :] = maxLine
    return outputH0CPU

def getMaxPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "MaxPlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def buildEngine(logger, inDatatype):
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        config.flags = int(inDatatype == np.float16)

    inputT0 = network.add_input("inputT0", npToTrt[inDatatype], [-1, -1, -1])
    profile.set_shape(inputT0.name, [1, 1, 1], [4, 3, 5], [9, 12, 6])
    inputT1 = network.add_input("inputT1", trt.int32, [-1])
    profile.set_shape(inputT1.name, [1], [4], [9])
    config.add_optimization_profile(profile)
    pluginLayer = network.add_plugin_v2([inputT0, inputT1], getMaxPlugin())

    network.mark_output(pluginLayer.get_output(0))
    return builder.build_engine(network, config)

def run(inDim, inDatatype):
    print("test", inDim, inDatatype)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger, inDatatype)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, inDim)
    context.set_binding_shape(1, inDim[:1])

    #print("Bind0->",engine.get_binding_shape(0),context.get_binding_shape(0))
    #print("Bind1->",engine.get_binding_shape(1),context.get_binding_shape(1))
    #print("Bind2->",engine.get_binding_shape(2),context.get_binding_shape(2))
    print("All bind:", context.all_binding_shapes_specified)
    stream = cuda.Stream()

    data0 = np.arange(np.prod(inDim), dtype=inDatatype).reshape(inDim)
    data1 = np.arange(1, inDim[0] + 1, dtype=np.int32)
    data1[data1 > inDim[1]] = inDim[1]
    inputH0 = np.ascontiguousarray(data0)
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = np.ascontiguousarray(data1)
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)

    stream.synchronize()

    outputH0CPU = maxCPU(inputH0, inputH1)

    #print("InputH0->",inputH0.shape, engine.get_binding_dtype(0))
    #print(inputH0)
    #print("InputH1->",inputH1.shape, engine.get_binding_dtype(1))
    #print(inputH1)
    #print("OutputH0->",outputH0.shape, engine.get_binding_dtype(2))
    #print(outputH0)
    #print("OutputH0CPU->",outputH0CPU.shape)
    #print(outputH0CPU)
    print("Check result:", ["True" if np.all(outputH0 == outputH0CPU) else "False"][0])

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cuda.Device(0).make_context()

    run([4, 3, 5], np.float32)
    run([9, 12, 6], np.float32)
    run([4, 3, 5], np.float16)
    run([9, 12, 6], np.float16)

    cuda.Context.pop()
    print("test finish!")
