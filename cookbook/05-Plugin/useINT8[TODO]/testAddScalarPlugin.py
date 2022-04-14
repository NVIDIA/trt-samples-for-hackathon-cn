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
from calibrator import MyCalibrator

soFile = "./AddScalarPlugin.so"
cacheFile = "./int8.cache"
epsilon = 1.0e-3
np.random.seed(97)

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def check(a, b, weak=False):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:", res, "maxAbsDiff:", diff0, "maxRelDiff:", diff1)

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'AddScalar':
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape, scalar):
    testCase = "<shape%s,scalar=%f>" % (shape, scalar)
    trtFile = "./model-Dims" + str(len(shape)) + ".plan"
    print("\nTest", testCase)
    logger = trt.Logger(trt.Logger.VERBOSE)
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
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        #config.max_workspace_size = 6 << 30
        #config.flags = (1 << int(trt.BuilderFlag.INT8)) | (1 << int(trt.BuilderFlag.STRICT_TYPES)) | (1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS))
        #config.flags = 1 << int(trt.BuilderFlag.FP16)
        config.flags = 1 << int(trt.BuilderFlag.INT8)
        #config.int8_calibrator = MyCalibrator(1,shape,cacheFile)

        inputT0 = network.add_input('inputT0', trt.float32, [-1,-1])
        inputT0.dynamic_range = [-100,100]
        profile.set_shape(inputT0.name, [1,1], [32,32], [64,64])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))
        pluginLayer.precision = trt.int8
        pluginLayer.set_output_type(0,trt.int8)
        pluginLayer.get_output(0).dtype = trt.int8
        pluginLayer.get_output(0).allowed_formats = (1<<int(trt.TensorFormat.CHW4))

        pluginLayer.get_output(0).dynamic_range = [-120,120]
        identityayer = network.add_identity(pluginLayer.get_output(0))
        identityayer.get_output(0).dynamic_range = [-120,120]
        identityayer.get_output(0).dtype = trt.float32

        #network.mark_output(pluginLayer.get_output(0))
        network.mark_output(identityayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        #with open(trtFile, 'wb') as f:
        #    f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, shape)
    _, stream = cudart.cudaStreamCreate()
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
                engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))

    bufferH = []
    data = np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 200 - 100
    bufferH.append(data)
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMallocAsync(bufferH[i].nbytes, stream)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)

    outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    check(bufferH[nInput], outputCPU[0], True)

    print(bufferH[0].reshape(-1)[:10])
    print(bufferH[-1].reshape(-1)[:10])

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test", testCase, "finish!")

if __name__ == '__main__':
    os.system('rm ./*.plan ./*.cache')
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    #run([512], 40)
    run([32, 32], 20)
    #run([16, 16, 16], 40)
    #run([8, 8, 8, 8], 40)

    print("test finish!")