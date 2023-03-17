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
from time import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath = "./SignPlugin.so"
np.random.seed(31193)

def reverseCPU(inputH0):
    return None

def getSignPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "SignPlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def buildEngine(logger, shape):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()

    inputT0 = network.add_input("inputT0", trt.float32, shape)
    oneHotLayer = network.add_plugin_v2([inputT0], getSignPlugin())

    network.mark_output(oneHotLayer.get_output(0))
    return builder.build_cuda_engine(network)

def run(batchSize, shape):
    print("test", batchSize, *shape)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger, shape)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()

    data = np.array(np.random.rand(batchSize, *shape) * 2 - 1, dtype=np.float32)
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty((batchSize, ) + tuple(context.get_binding_shape(1)), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(batchSize, [int(inputD0), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    #print("data:", np.shape(data), data.dtype, np.mean(data), np.var(data), np.max(data), np.min(data))
    #print(data)
    #print("hOut:", np.shape(outputH0), outputH0.dtype, np.mean(outputH0), np.var(outputH0), np.max(outputH0), np.min(outputH0))
    #print(outputH0)
    print("check result:", np.all(np.sign(data) == outputH0), "\n")

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run(4, [16])
    run(4, [18])
    run(4, [600])
    print("test finish!")
