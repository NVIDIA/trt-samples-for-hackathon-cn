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
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

npToNumber = {np.float32: 0, np.float16: 1, np.int8: 2, np.int32: 3}
soFilePath = "./TopKAveragePlugin.so"

def topKAverageCPU(inputH0, inputH1, inputH2, inputH3):
    sh = inputH0.shape
    nTopK = len(inputH3)
    outputH0CPU = np.zeros([sh[0], sh[2], sh[1] * len(inputH3)], dtype=np.float32)
    for i in range(sh[0]):
        data = np.sort(inputH0[i, :, :inputH1[i], :inputH2[i]])
        for k in range(nTopK):
            outputH0CPU[i, :inputH1[i], k::nTopK] = np.sum(data[:, :, -inputH3[k]:], axis=2).transpose() / inputH3[k]
    return outputH0CPU

def cleanTrash(outputH0, inputH1):  # clean the trash data in the output of GPU
    for i in range(outputH0.shape[0]):
        outputH0[i, inputH1[i]:, :] = 0
    return outputH0

def getTopKAveragePlugin(nTopK, maxTopK):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "TopKAveragePlugin":
            p0 = trt.PluginField("nTopK", np.array([nTopK], dtype=np.int32), trt.PluginFieldType.INT32)
            p1 = trt.PluginField("maxTopK", np.array([maxTopK], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin("TopKAveragePlugin", trt.PluginFieldCollection([p0, p1]))
    return None

def buildEngine(logger, outDatatype, nTopK, maxTopK):
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        config.flags = int(outDatatype == np.float16)

    inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
    profile.set_shape(inputT0.name, [1, 1, 1, 1], [36, 10, 5, 30], [72, 20, 10, 70])
    inputT1 = network.add_input("inputT1", trt.int32, [-1])
    profile.set_shape(inputT1.name, [1], [36], [72])
    inputT2 = network.add_input("inputT2", trt.int32, [-1])
    profile.set_shape(inputT2.name, [1], [36], [72])
    inputT3 = network.add_input("inputT3", trt.int32, [-1])
    profile.set_shape(inputT3.name, [1], [2], [4])

    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2([inputT0, inputT1, inputT2, inputT3], getTopKAveragePlugin(nTopK, maxTopK))

    network.mark_output(pluginLayer.get_output(0))
    return builder.build_engine(network, config)

def run(inDim, outDatatype, topKList):
    print("test", inDim, outDatatype, topKList)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger, outDatatype, len(topKList), max(topKList))
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, inDim)
    context.set_binding_shape(1, inDim[:1])
    context.set_binding_shape(2, inDim[:1])
    context.set_binding_shape(3, [len(topKList)])

    #print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))
    #print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))
    #print("Bind2->", engine.get_binding_shape(2), context.get_binding_shape(2))
    #print("Bind3->", engine.get_binding_shape(3), context.get_binding_shape(3))
    #print("Bind4->", engine.get_binding_shape(4), context.get_binding_shape(4))
    print("All bind:", context.all_binding_shapes_specified)
    stream = cuda.Stream()

    data0 = np.tile(np.arange(1, 1 + np.prod(inDim[-2:]), dtype=np.float32).reshape(inDim[-2:]), [*inDim[:2], 1, 1])
    data1 = np.arange(inDim[0], dtype=np.int32) % inDim[2] + 1
    data2 = np.arange(inDim[0], dtype=np.int32) % inDim[3] + 1
    data3 = np.array(topKList, dtype=np.int32)
    inputH0 = np.ascontiguousarray(data0)
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = np.ascontiguousarray(data1)
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    inputH2 = np.ascontiguousarray(data2)
    inputD2 = cuda.mem_alloc(inputH2.nbytes)
    inputH3 = np.ascontiguousarray(data3)
    inputD3 = cuda.mem_alloc(inputH3.nbytes)
    outputH0 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    cuda.memcpy_htod_async(inputD2, inputH2, stream)
    cuda.memcpy_htod_async(inputD3, inputH3, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(inputD3), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)

    stream.synchronize()

    outputH0CPU = topKAverageCPU(inputH0, inputH1, inputH2, inputH3)

    #print("Input0:",inputH0.shape,engine.get_binding_dtype(0))
    #print(inputH0)
    #print("Input1:",inputH1.shape,engine.get_binding_dtype(1))
    #print(inputH1)
    #print("Input2:",inputH2.shape,engine.get_binding_dtype(2))
    #print(inputH2)
    #print("Input3:",inputH3.shape,engine.get_binding_dtype(3))
    #print(inputH3)
    #print("Output:",outputH0.shape, engine.get_binding_dtype(4))
    #print(outputH0)
    print("Check result:", ["True" if np.all(cleanTrash(outputH0, inputH1) == outputH0CPU) else "False"][0])

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    np.set_printoptions(threshold=1e6)
    cuda.Device(0).make_context()

    run((36, 10, 5, 30), np.float32, [2, 3, 4])
    run((36, 8, 5, 65), np.float32, [1, 2, 5, 12])
    run((36, 18, 5, 70), np.float32, [1, 2, 5, 12])

    cuda.Context.pop()
    print("test finish!")
