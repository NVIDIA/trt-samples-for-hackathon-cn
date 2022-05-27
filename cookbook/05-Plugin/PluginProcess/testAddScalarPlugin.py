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
from cuda import cudart
import tensorrt as trt

soFile = "./AddScalarPlugin.so"
epsilon = 1.0e-2
np.random.seed(97)

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'AddScalar':
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape0, shape1, scalar):
    testCase = "<shape0:%s,shape1:%s,scalar=%f>" % (shape0,shape1, scalar)
    trtFile = "./model-Dims" + str(len(shape0)) + ".plan"
    print("\nTest", testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile0 = builder.create_optimization_profile()
        profile1 = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 6 << 30
        config.flags = 1 << int(trt.BuilderFlag.FP16)  # 注释掉这一行，Pugin 就仅使用 FP32

        inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, [-1 for i in shape0])
        profile0.set_shape(inputT0.name, [1 for i in shape0], [8 for i in shape0], [32 for i in shape0])
        config.add_optimization_profile(profile0)
        profile1.set_shape(inputT0.name, [1 for i in shape1], [8 for i in shape1], [32 for i in shape1])
        config.add_optimization_profile(profile1)

        pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))

        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    stream = 0  # 使用默认 CUDA 流
    cudart.cudaStreamSynchronize(stream)

    # 使用 Profile 0
    print("Use Profile 0")
    context.set_optimization_profile_async(0, stream)
    cudart.cudaStreamSynchronize(stream)
    #context.active_optimization_profile = 0  # 与上面两行等价的选择 profile 的方法，不需要用 stream，但是将被废弃
    context.set_binding_shape(0, shape0)
    print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    for i in range(engine.num_bindings):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.random.rand(np.prod(shape0)).reshape(shape0).astype(np.float32)*2-1
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
    _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    print("before inference")
    context.execute_v2([int(inputD0), int(outputD0), int(0), int(0)])
    print("after inference")
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # 使用 Profile 1
    print("Use Profile 1")
    context.set_optimization_profile_async(1, stream)
    cudart.cudaStreamSynchronize(stream)
    #context.active_optimization_profile = 1  # 与上面两行等价的选择 profile 的方法，不需要用 stream，但是将被废弃
    context.set_binding_shape(2, shape1)
    print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    for i in range(engine.num_bindings):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.random.rand(np.prod(shape1)).reshape(shape1).astype(np.float32)*2-1
    inputH1 = np.ascontiguousarray(data.reshape(-1))
    outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
    _, inputD1 = cudart.cudaMalloc(inputH1.nbytes)
    _, outputD1 = cudart.cudaMalloc(outputH1.nbytes)

    cudart.cudaMemcpy(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    print("before inference")
    context.execute_v2([int(0), int(0), int(inputD1), int(outputD1)])
    print("after inference")
    cudart.cudaMemcpy(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)

if __name__ == '__main__':
    os.system('rm ./*.plan')
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    run([8, 8], [32, 32], 1)
    run([8, 8], [32, 32], 1)

    print("test finish!")
