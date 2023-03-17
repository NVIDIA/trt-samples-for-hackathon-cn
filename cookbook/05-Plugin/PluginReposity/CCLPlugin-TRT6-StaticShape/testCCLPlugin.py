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

soFilePath = "./CCLPlugin.so"
height = 384
width = 640
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
    builder.max_batch_size = 1
    builder.set_memory_pool_limit = 3 << 30
    builder.fp16_mode = False
    network = builder.create_network()

    inputT0 = network.add_input("pixelScore", trt.float32, (height, width))
    inputT1 = network.add_input("linkScore", trt.float32, (8, height, width))
    cclLayer = network.add_plugin_v2([inputT0, inputT1], getCCLPlugin())

    network.mark_output(cclLayer.get_output(0))
    network.mark_output(cclLayer.get_output(1))
    return builder.build_cuda_engine(network)

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()
    inputH0 = np.ascontiguousarray(np.random.rand(height, width).reshape(-1))
    inputH1 = np.ascontiguousarray(np.random.rand(8, height, width).reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    outputH1 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputD1 = cuda.mem_alloc(outputH1.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    stream.synchronize()
    context.execute_async(1, [int(inputD0), int(inputD1), int(outputD0), int(outputD1)], stream.handle)
    stream.synchronize()
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    print(np.shape(outputH0), np.shape(outputH1))
    #print(outputH0)
    #print(outputH1)

if __name__ == "__main__":
    run()
    print("test finish!")
