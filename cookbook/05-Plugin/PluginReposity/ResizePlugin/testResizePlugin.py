#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

soFilePath = "./ResizePlugin.so"
hOut = 5
wOut = 9

def getResizePlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'ResizePlugin':
            p0 = trt.PluginField("hOut", np.array([hOut], dtype=np.int32), trt.PluginFieldType.INT32)
            p1 = trt.PluginField("wOut", np.array([wOut], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0, p1]))
    return None

def buildEngine(logger):
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()

    inputTensor = network.add_input("inputT0", trt.float32, (2, 3, 4))
    resizeLayer = network.add_plugin_v2([inputTensor], getResizePlugin())
    network.mark_output(resizeLayer.get_output(0))
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
    data = np.array([7, 5, 6, 4, 4, 2, 5, 3, 3, 9, 9, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(2, 3, 4).astype(np.float32)
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(1, [int(inputD0), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    print("input=\n", data)
    print("real output=\n", outputH0)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    run()
    print("test finish!")
