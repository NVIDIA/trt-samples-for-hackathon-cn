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

soFilePath = "./SortPlugin.so"
np.random.seed(31193)
epsilon = 1e-6
nElement = 1024
nWidth = 1

def sortCPU(inputH0, inputH1):
    index = np.lexsort((inputH1, inputH0))
    output = np.array([[inputH0[index[i]], inputH1[index[i]]] for i in range(1024)])
    return output

def getSortPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "SortPlugin":
            p0 = trt.PluginField("descending", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def buildEngine(logger):
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network()

    tensor1 = network.add_input("dataKey", trt.float32, (nElement, 1))
    tensor2 = network.add_input("dataValue", trt.float32, (nElement, nWidth))
    sortLayer = network.add_plugin_v2([tensor1, tensor2], getSortPlugin())

    network.mark_output(sortLayer.get_output(0))
    network.mark_output(sortLayer.get_output(1))

    return builder.build_engine(network, config)

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

    inputH0 = np.ascontiguousarray(np.random.rand(nElement).astype(np.float32).reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputH1 = np.ascontiguousarray(np.random.rand(nElement, nWidth).astype(np.float32).reshape(-1))
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    outputH0 = np.empty(engine.get_binding_shape(2), dtype=np.float32)
    outputD0 = cuda.mem_alloc(outputH0.nbytes)
    outputH1 = np.empty(engine.get_binding_shape(3), dtype=np.float32)
    outputD1 = cuda.mem_alloc(outputH1.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    context.execute_async(1, [int(inputD0), int(inputD1), int(outputD0), int(outputD1)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    cuda.memcpy_dtoh_async(outputH1, outputD1, stream)
    stream.synchronize()

    outputCPU = sortCPU(inputH0, inputH1)

    print(np.shape(outputH0), np.shape(outputH1))
    print("Check result Key:", "True" if np.mean(np.abs(outputH0.reshape(-1) - outputCPU[:, 0].reshape(-1))) < epsilon else "False")
    print("Check result Value:", "True" if np.mean(np.abs(outputH1.reshape(-1) - outputCPU[:, 1].reshape(-1))) < epsilon else "False")
    """
    for i in range(1000):
        print("%4d"%i,(inputH0[i],inputH1[i]),outputCPU[i],outputH0[i],outputH1[i])
    """

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run()
    print("test finish!")
