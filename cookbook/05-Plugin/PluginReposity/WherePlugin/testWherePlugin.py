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

soFilePath = "./WherePlugin.so"
usingFp16 = False

def whereCPU(condition, inputX, inputY):
    return inputX * condition + inputY * (1 - condition)

def getWherePlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "WherePlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def buildEngine(logger, nRow, nCol):
    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30
    builder.fp16_mode = usingFp16
    network = builder.create_network()

    tensor1 = network.add_input("condition", trt.int32, (nRow, nCol))
    tensor2 = network.add_input("inputX", trt.float32, (nRow, nCol))
    tensor3 = network.add_input("inputY", trt.float32, (nRow, nCol))
    whereLayer = network.add_plugin_v2([tensor1, tensor2, tensor3], getWherePlugin())

    network.mark_output(whereLayer.get_output(0))
    return builder.build_cuda_engine(network)

def run(batchSize, nRow, nCol):
    print("test", batchSize, nRow, nCol)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger, nRow, nCol)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeeded building engine!")

    context = engine.create_execution_context()
    stream = cuda.Stream()

    condition = np.array(np.random.randint(0, 2, [batchSize, nRow, nCol]), dtype=np.int32)
    inputX = np.full([batchSize, nRow, nCol], 1, dtype=np.float32)
    inputY = np.full([batchSize, nRow, nCol], -1, dtype=np.float32)
    inputH0 = np.ascontiguousarray(condition.reshape(-1))
    inputH1 = np.ascontiguousarray(inputX.reshape(-1))
    inputH2 = np.ascontiguousarray(inputY.reshape(-1))
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    inputD1 = cuda.mem_alloc(inputH1.nbytes)
    inputD2 = cuda.mem_alloc(inputH2.nbytes)
    outputH0 = np.empty((batchSize, ) + tuple(engine.get_binding_shape(3)), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    cuda.memcpy_htod_async(inputD1, inputH1, stream)
    cuda.memcpy_htod_async(inputD2, inputH2, stream)
    context.execute_async(batchSize, [int(inputD0), int(inputD1), int(inputD2), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()

    outputH0CPU = whereCPU(condition, inputX, inputY)
    print("Check result:", ["True" if np.all(outputH0 == outputH0CPU) else "False"][0])

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run(4, 5, 4)
    run(4, 20, 9)
    run(4, 200, 10)
    print("test finish!")
