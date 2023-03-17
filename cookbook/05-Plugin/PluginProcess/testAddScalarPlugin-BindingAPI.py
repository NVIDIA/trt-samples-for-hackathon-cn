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
from cuda import cudart
import numpy as np
import os
import tensorrt as trt

soFile = "./AddScalarPlugin.so"
nProfile = 2
np.set_printoptions(precision=3, linewidth=100, suppress=True)
np.random.seed(31193)
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

def run(shape, scalar, bFP16):
    testCase = "<shape=%s,scalar=%f,FP16=%s>" % (shape, scalar, bFP16)
    trtFile = "./model-Dim%s-FP%s.plan" % (str(len(shape)), ("16" if bFP16 else "32"))
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

        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
        for k, profile in enumerate(profileList):
            profile.set_shape(inputT0.name, [1*(k+1) for i in shape], [2*(k+1) for i in shape], [4*(k+1) for i in shape])
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

    cudaStreamList = [int(cudart.cudaStreamCreate()[1]) for i in range(nProfile)]
    nIO = engine.num_bindings - nInput
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = nIO - nInput
    nInput = nInput // nProfile
    nOutput = nOutput // nProfile

    context = engine.create_execution_context()

    bufferH = []
    for index in range(nProfile):
        context.set_optimization_profile_async(index, cudaStreamList[index])
        bindingPad = (nInput + nOutput) * index  # skip bindings of previous optimization profile
        bindingShape = (np.array(shape) * (index + 1)).tolist()
        context.set_binding_shape(bindingPad + 0, bindingShape)
        print("Context%d binding all? %s" % (index, "Yes" if context.all_binding_shapes_specified else "No"))
        for i in range(engine.num_bindings):
            print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))
        for i in range(nInput):
            bufferH.append(np.arange(np.prod(bindingShape)).astype(np.float32).reshape(bindingShape))
        for i in range(nOutput):
            bufferH.append(np.empty(context.get_binding_shape(bindingPad + nInput + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for index in range(nProfile):
        print("Use Profile %d" % index)
        context.set_optimization_profile_async(index, cudaStreamList[index])  # 重设 Profile 后需要重新绑定输入张量形状
        bindingPad = (nInput + nOutput) * index
        bindingShape = (np.array(shape) * (index + 1)).tolist()
        context.set_binding_shape(bindingPad + 0, bindingShape)

        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, cudaStreamList[index])

        bufferList = [int(0) for b in bufferD[:bindingPad]] + [int(b) for b in bufferD[bindingPad:(bindingPad + nInput + nOutput)]] + [int(0) for b in bufferD[(bindingPad + nInput + nOutput):]]
        # Fill int(0) beside the bindings to the context now using, bufferList = bufferD is also OK

        context.execute_async_v2(bufferList, cudaStreamList[index])

    for index in range(nProfile):
        bindingPad = (nInput + nOutput) * index
        for i in range(nOutput):
            cudart.cudaMemcpyAsync(bufferH[bindingPad + nInput + i].ctypes.data, bufferD[bindingPad + nInput + i], bufferH[bindingPad + nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, cudaStreamList[index])

    for index in range(nProfile):
        cudart.cudaStreamSynchronize(cudaStreamList[index])

    for index in range(nProfile):
        bindingPad = (nInput + nOutput) * index
        print("check Profile %d:" % index)
        check(bufferH[bindingPad + 1], bufferH[bindingPad + 0] + 1, True)

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    
    run([4, 4], 1, False)
    run([4, 4], 1, False)

    run([4, 4], 1, True)
    run([4, 4], 1, True)

    print("Test all finish!")
