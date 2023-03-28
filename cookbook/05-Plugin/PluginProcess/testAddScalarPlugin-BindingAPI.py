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

# For TensorRT < 8.5 with deprecated Binding API

import ctypes
import os

import numpy as np
import tensorrt as trt
from cuda import cudart

soFile = "./AddScalarPlugin.so"
nProfile = 2
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(bFP16):
    shapeSmall = [2, 4, 4, 4]
    scalar = 1
    testCase = "<FP16=%s>" % bFP16
    trtFile = "./model-FP%s.plan" % ("16" if bFP16 else "32")
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profileList = [builder.create_optimization_profile() for index in range(nProfile)]
        config = builder.create_builder_config()
        if bFP16:
            config.set_flag(trt.BuilderFlag.FP16)

        inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
        for profile in profileList:
            profile.set_shape(inputT0.name, shapeSmall, shapeSmall, (np.array(shapeSmall) * 2).tolist())
            config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))
        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_bindings
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = nIO - nInput
    nIO = nIO // nProfile
    nInput = nInput // nProfile
    nOutput = nOutput // nProfile

    cudaStreamList = [int(cudart.cudaStreamCreate()[1]) for i in range(nProfile)]
    context = engine.create_execution_context()

    bufferH = []  # use respective buffers for different Optimization Profile
    for index in range(nProfile):
        context.set_optimization_profile_async(index, cudaStreamList[index])
        bindingPad = nIO * index  # skip bindings previous OptimizationProfile occupy
        shape = (np.array(shapeSmall) * (index + 1)).tolist()  # use different shapes
        context.set_binding_shape(bindingPad + 0, shape)
        for i in range(nInput):
            bufferH.append(np.arange(np.prod(shape)).astype(np.float32).reshape(shape))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_binding_shape(bindingPad + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + i))))
    bufferD = []
    for i in range(len(bufferH)):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for index in range(nProfile):
        print("Use Profile %d" % index)
        context.set_optimization_profile_async(index, cudaStreamList[index])  # set shape again after changing the optimization profile
        bindingPad = nIO * index
        shape = (np.array(shapeSmall) * (index + 1)).tolist()
        context.set_binding_shape(bindingPad + 0, shape)
        for i in range(nIO * nProfile):
            print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, cudaStreamList[index])

        context.execute_async_v2(bufferD, cudaStreamList[index])

        for i in range(nInput, nIO):
            cudart.cudaMemcpyAsync(bufferH[bindingPad + i].ctypes.data, bufferD[bindingPad + i], bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, cudaStreamList[index])

        cudart.cudaStreamSynchronize(cudaStreamList[index])

    for index in range(nProfile):
        bindingPad = nIO * index
        print("check OptimizationProfile %d:" % index)
        check(bufferH[bindingPad + 1], bufferH[bindingPad + 0] + 1, True)

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")

    run(False)
    run(False)

    run(True)
    run(True)

    print("Test all finish!")
