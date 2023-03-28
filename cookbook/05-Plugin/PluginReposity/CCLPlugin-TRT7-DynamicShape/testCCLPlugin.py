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
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

#import matplotlib.pyplot as plt

soFilePath = "./CCLPlugin.so"
np.random.seed(31193)

def getCCLPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "CCLPlugin":
            p0 = trt.PluginField("minPixelScore", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p1 = trt.PluginField("minLinkScore", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p2 = trt.PluginField("minArea", np.array([10], dtype=np.int32), trt.PluginFieldType.INT32)
            p3 = trt.PluginField("maxcomponentCount", np.array([65536], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0, p1, p2, p3]))
    return None

def buildEngine(logger):
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    inputT0 = network.add_input("pixelScore", trt.float32, (-1, -1, -1))
    profile.set_shape(inputT0.name, [1, 1, 1], [2, 384, 640], [4, 768, 1280])
    inputT1 = network.add_input("linkScore", trt.float32, (-1, 8, -1, -1))
    profile.set_shape(inputT1.name, [1, 8, 1, 1], [4, 8, 384, 640], [8, 8, 768, 1280])
    config.add_optimization_profile(profile)

    cclLayer = network.add_plugin_v2([inputT0, inputT1], getCCLPlugin())

    network.mark_output(cclLayer.get_output(0))
    network.mark_output(cclLayer.get_output(1))
    return builder.build_engine(network, config)

def run(inDim):
    print("test", inDim)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, inDim)
    context.set_binding_shape(1, inDim[:1] + [8] + inDim[1:])
    stream = cuda.Stream()

    data0 = np.random.rand(np.prod(inDim)).reshape(-1)
    data1 = np.random.rand(np.prod(inDim) * 8).reshape(-1)
    inputH0 = np.ascontiguousarray(data0)
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = np.ascontiguousarray(data1)
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputH1 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputD1 = cuda.mem_alloc(outputH1.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    stream.synchronize()
    context.execute_async_v2([int(inputD0), int(inputD1), int(outputD0), int(outputD1)], stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    print(np.shape(outputH0), np.shape(outputH1))
    #print(outputH0)
    #print(outputH1)
    #plt.imshow(outputH0/np.max(outputH0))
    #plt.show()

if __name__ == "__main__":
    run([1, 1, 1])
    run([2, 384, 640])
    run([4, 768, 1280])
    print("test finish!")
