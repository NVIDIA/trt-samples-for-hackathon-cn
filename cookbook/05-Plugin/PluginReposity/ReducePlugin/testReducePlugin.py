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

soFilePath = "./ReducePlugin.so"
np.random.seed(31193)

def reduceCPU(inputH0, isSum):
    if isSum:
        return np.sum(inputH0, -2)
    else:
        return np.max(inputH0, -2)

def getReducePlugin(isSum):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "ReducePlugin":
            p0 = trt.PluginField("isSum", np.array([int(isSum)], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def buildEngine(logger, shape, isSum):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()

    inputTensor = network.add_input("inputT0", trt.float32, shape)
    reduceLayer = network.add_plugin_v2([inputTensor], getReducePlugin(isSum))
    network.mark_output(reduceLayer.get_output(0))
    return builder.build_cuda_engine(network)

def run(nBatchSize, shape, isSum):
    print("test", nBatchSize, shape, isSum)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)
    engine = buildEngine(logger, shape, isSum)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()
    data = np.random.rand(*[nBatchSize, *shape]).astype(np.float32)
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty((nBatchSize, ) + tuple(context.get_binding_shape(1)), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outpuD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(nBatchSize, [int(inputD0), int(outpuD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outpuD0, stream)
    stream.synchronize()
    outputH0CPU = reduceCPU(data, isSum)

    print("Check result:", ["True" if np.all(outputH0 == outputH0CPU) else "False"][0])
    """
    temp = outputH0
    print(temp.shape, temp.dtype, np.mean(temp), np.var(temp), np.max(temp), np.min(temp))
    print(temp)
    temp = outputH0CPU
    print(temp.shape, temp.dtype, np.mean(temp), np.var(temp), np.max(temp), np.min(temp))
    print(temp)
    """

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run(4, [8, 2, 128], False)
    run(4, [8, 5, 128], False)
    run(4, [8, 6, 128], False)
    run(4, [8, 10, 128], False)
    run(4, [8, 15, 128], False)
    run(4, [8, 16, 128], False)
    run(4, [8, 30, 128], False)
    run(4, [8, 82, 128], False)
    run(4, [8, 30, 128], True)
    print("test finish!")
