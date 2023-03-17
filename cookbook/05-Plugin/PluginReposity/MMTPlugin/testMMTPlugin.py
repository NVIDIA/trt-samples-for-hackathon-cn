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

npToTrt = {np.float32: trt.float32, np.float16: trt.float16}
soFilePath = "./MMTPlugin.so"

def MMTCPU(inputH0, inputH1, weight):
    sh0 = inputH0.shape
    sh1 = inputH1.shape
    h, dim_t, _ = weight.shape
    outputCPU = np.zeros([sh0[0], dim_t, sh0[1], sh1[1]], dtype=np.float32)
    for i in range(sh0[0]):
        outputCPU[i] = np.matmul(np.matmul(inputH0[0], weight.transpose(0, 2, 1)).transpose(2, 1, 0), inputH1[0].transpose())
    return outputCPU

def getMMTPlugin(h, dim_t, weight):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "MMTPlugin":
            p0 = trt.PluginField("w", np.array([weight], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            p1 = trt.PluginField("h", np.array([h], dtype=np.int32), trt.PluginFieldType.INT32)
            p2 = trt.PluginField("dim_t", np.array([dim_t], dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0, p1, p2]))
    return None

def buildEngine(logger, shape, dim_t, weight, datatype):
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        config.flags = int(datatype == np.float16)

    inputT0 = network.add_input("x", npToTrt[datatype], (-1, -1, -1))
    profile.set_shape(inputT0.name, (1, 1, 1), shape, [i * 2 for i in shape])
    inputT1 = network.add_input("y", npToTrt[datatype], (-1, -1, -1))
    profile.set_shape(inputT1.name, (1, 1, 1), shape, [i * 2 for i in shape])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2([inputT0, inputT1], getMMTPlugin(shape[-1], dim_t, weight))

    network.mark_output(pluginLayer.get_output(0))
    return builder.build_engine(network, config)

def run(nGroup, xWidth, yWidth, h, dim_t, datatype):
    print("test [%d,%d/%d,%d],dim_t=%d" % (nGroup, xWidth, yWidth, h, dim_t), datatype)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)
    weight = np.full([h, dim_t, h], 0.1, dtype=np.float32)
    engine = buildEngine(logger, [nGroup, max(xWidth, yWidth), h], dim_t, weight, datatype)
    if engine == None:
        print("Failed building engine!")
        return None
    print("Succeed building engine!")

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nGroup, xWidth, h])
    context.set_binding_shape(1, [nGroup, yWidth, h])
    #print("Binding0->",engine.get_binding_shape(0),context.get_binding_shape(0))
    #print("Binding1->",engine.get_binding_shape(1),context.get_binding_shape(1))
    #print("Binding2->",engine.get_binding_shape(2),context.get_binding_shape(2))
    #print("All bind:",context.all_binding_shapes_specified)
    stream = cuda.Stream()

    data0 = np.ones([nGroup, xWidth, h], dtype=datatype)
    data1 = np.ones([nGroup, yWidth, h], dtype=datatype)
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

    outputH0CPU = MMTCPU(inputH0, inputH1, weight)

    #print("InputH0->",inputH0.shape, engine.get_binding_dtype(0))
    #print(inputH0)
    #print("InputH1->",inputH1.shape, engine.get_binding_dtype(1))
    #print(inputH1)
    #print("OutputH0->",outputH0.shape, engine.get_binding_dtype(2))
    #print(outputH0)
    print("Check result:", ["True" if np.all(outputH0 == outputH0CPU) else "False"][0])

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    cuda.Device(0).make_context()

    run(4, 5, 6, 2, 3, np.float32)
    run(4, 5, 6, 2, 3, np.float16)

    cuda.Context.pop()
    print("test finish!")
