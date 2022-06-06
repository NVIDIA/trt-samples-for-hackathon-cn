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

import os
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

npToTrt = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}
soFilePath = "./CumSumPlugin.so"

def cumSumCPU(inputH0, axis):
    return np.cumsum(inputH0, axis)

def getCumSumPlugin(axis):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'CumSumPlugin':
            p0 = trt.PluginField("axis", np.array([axis], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

def buildEngine(logger, nInDim, inDatatype, axis):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << 0)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 7 << 30
    config.flags = int(inDatatype == np.float16)

    if nInDim == 1:
        inputT0 = network.add_input('input0', npToTrt[inDatatype], [-1])
        profile.set_shape(inputT0.name, [1], [32], [1024])
    elif nInDim == 2:
        inputT0 = network.add_input('input0', npToTrt[inDatatype], [-1, -1])
        profile.set_shape(inputT0.name, [1, 1], [32, 32], [256, 256])
    elif nInDim == 3:
        inputT0 = network.add_input('input0', npToTrt[inDatatype], [-1, -1, -1])
        profile.set_shape(inputT0.name, [1, 1, 1], [32, 32, 32], [256, 256, 256])
    elif nInDim == 4:
        inputT0 = network.add_input('input0', npToTrt[inDatatype], [-1, -1, -1, -1])
        profile.set_shape(inputT0.name, [1, 1, 1, 1], [32, 32, 32, 32], [256, 256, 256, 256])
    else:
        print("Error in buildEngine, nInDim == %d" % nInDim)
        return None

    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2([inputT0], getCumSumPlugin(axis))

    network.mark_output(pluginLayer.get_output(0))

    return builder.build_engine(network, config)

def run(inDim, inDatatype, inAxis):
    print("test", inDim, inDatatype, "axis=%d" % inAxis)
    errorLimit = 1e-3 if inDatatype == np.float16 else 1e-6

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    engine = buildEngine(logger, len(inDim), inDatatype, inAxis)
    if engine == None:
        print("Failed building engine!")
        return None
    #print("succeeded building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, inDim)
    stream = cuda.Stream()

    data0 = np.arange(np.prod(inDim), dtype=inDatatype).reshape(inDim)
    inputH0 = np.ascontiguousarray(data0)
    inputD0 = cuda.mem_alloc(inputH0.nbytes)
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    outputD0 = cuda.mem_alloc(outputH0.nbytes)

    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)

    stream.synchronize()

    outputH0CPU = cumSumCPU(inputH0, inAxis)
    '''
    print("InputH0->",inputH0.shape, engine.get_binding_dtype(0))
    print(inputH0)
    print("OutputH0->",outputH0.shape, engine.get_binding_dtype(1))
    print(outputH0)
    print("OutputH0CPU->",outputH0CPU.shape)
    print(outputH0CPU)
    '''
    #print(np.mean( outputH0 - outputH0CPU ))
    print("Check result:", ["True" if np.mean(outputH0 - outputH0CPU) < errorLimit else "False"][0])

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    cuda.Device(0).make_context()

    # w 维
    run([16], np.float32, 0)
    run([16], np.float16, 0)
    run([16], np.int32, 0)
    run([2, 16], np.float32, 1)
    run([2, 16], np.float16, 1)
    run([2, 16], np.int32, 1)
    run([2, 3, 16], np.float32, 2)
    run([2, 3, 16], np.float16, 2)
    run([2, 3, 16], np.int32, 2)
    run([2, 3, 4, 16], np.float32, 3)
    run([2, 3, 4, 16], np.float16, 3)
    run([2, 3, 4, 16], np.int32, 3)
    run([256], np.float32, 0)
    run([256], np.float16, 0)
    run([256], np.int32, 0)
    run([2, 256], np.float32, 1)
    run([2, 256], np.float16, 1)
    run([2, 256], np.int32, 1)
    run([2, 3, 256], np.float32, 2)
    run([2, 3, 256], np.float16, 2)  # 数据范围不足，产生 inf
    run([2, 3, 256], np.int32, 2)
    run([2, 3, 4, 256], np.float32, 3)
    run([2, 3, 4, 256], np.float16, 3)
    run([2, 3, 4, 256], np.int32, 3)

    # h 维
    run([2, 16], np.float32, 0)
    run([2, 16], np.float16, 0)
    run([2, 16], np.int32, 0)
    run([2, 3, 16], np.float32, 1)
    run([2, 3, 16], np.float16, 1)
    run([2, 3, 16], np.int32, 1)
    run([2, 3, 4, 16], np.float32, 2)
    run([2, 3, 4, 16], np.float16, 2)
    run([2, 3, 4, 16], np.int32, 2)

    run([2, 256], np.float32, 0)
    run([2, 256], np.float16, 0)
    run([2, 256], np.int32, 0)
    run([2, 3, 256], np.float32, 1)
    run([2, 3, 256], np.float16, 1)
    run([2, 3, 256], np.int32, 1)
    run([2, 3, 4, 256], np.float32, 2)
    run([2, 3, 4, 256], np.float16, 2)
    run([2, 3, 4, 256], np.int32, 2)

    # c 维
    run([2, 3, 16], np.float32, 0)
    run([2, 3, 16], np.float16, 0)
    run([2, 3, 16], np.int32, 0)
    run([2, 3, 4, 16], np.float32, 1)
    run([2, 3, 4, 16], np.float16, 1)
    run([2, 3, 4, 16], np.int32, 1)

    run([2, 3, 256], np.float32, 0)
    run([2, 3, 256], np.float16, 0)
    run([2, 3, 256], np.int32, 0)
    run([2, 3, 4, 256], np.float32, 1)
    run([2, 3, 4, 256], np.float16, 1)
    run([2, 3, 4, 256], np.int32, 1)

    # n 维
    run([2, 3, 4, 16], np.float32, 0)
    run([2, 3, 4, 16], np.float16, 0)
    run([2, 3, 4, 16], np.int32, 0)

    run([2, 3, 4, 256], np.float32, 0)
    run([2, 3, 4, 256], np.float16, 0)
    run([2, 3, 4, 256], np.int32, 0)

    cuda.Context.pop()
    print("test finish!")
