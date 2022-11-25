#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
np.random.seed(31193)

def printArrayInfomation(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "AddScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run(shape0, scalar):
    testCase = "<shape0:%s,scalar=%f>" % (shape0, scalar)
    trtFile = "./model-Dims" + str(len(shape0)) + ".plan"
    print("\nTest", testCase)
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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape0])
        for profile in profileList:
            profile.set_shape(inputT0.name, [1 for i in shape0], [8 for i in shape0], [32 for i in shape0])
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

    stream = 0  # 使用默认 CUDA 流
    context = engine.create_execution_context()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    nInput = nInput // nProfile
    nOutput = nOutput // nProfile

    bufferH = []
    for index in range(nProfile):
        context.set_optimization_profile_async(index, stream)
        bindingPad = (nInput + nOutput) * index  # 跳过前面 OptimizationProfile 占用的 Binding
        bindingShape = (np.array(shape0) * (index + 1)).tolist()
        context.set_binding_shape(bindingPad + 0, bindingShape)
        print("Context%d binding all? %s" % (index, "Yes" if context.all_binding_shapes_specified else "No"))
        for i in range(engine.num_bindings):
            print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

        for i in range(nInput):
            #bufferH.append(np.random.rand(*bindingShape).astype(np.float32) * 2 - 1)
            bufferH.append(np.arange(np.prod(bindingShape)).astype(np.float32).reshape(bindingShape))
        for i in range(nOutput):
            bufferH.append(np.empty(context.get_binding_shape(bindingPad + nInput + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + nInput + i))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for index in range(nProfile):
        bindingPad = (nInput + nOutput) * index
        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    for index in range(nProfile):
        print("Use Profile %d" % index)
        context.set_optimization_profile_async(index, stream)  # 重设 Profile 后需要重新绑定输入张量形状
        bindingPad = (nInput + nOutput) * index
        bindingShape = (np.array(shape0) * (index + 1)).tolist()
        context.set_binding_shape(bindingPad + 0, bindingShape)
        bindingPad = (nInput + nOutput) * index
        bufferList = [int(0) for b in bufferD[:bindingPad]] + [int(b) for b in bufferD[bindingPad:(bindingPad + nInput + nOutput)]] + [int(0) for b in bufferD[(bindingPad + nInput + nOutput):]]
        # 分为三段，除了本 Context 对应的 Binding 位置上的 bufferD 以外，全部填充 int(0)
        # 其实也可以直接 bufferList = bufferD，只不过除了本 Context 对应的 Binding 位置上的 bufferD 以外全都用不到
        context.execute_async_v2(bufferList, stream)

    for index in range(nProfile):
        bindingPad = (nInput + nOutput) * index
        for i in range(nOutput):
            cudart.cudaMemcpyAsync(bufferH[bindingPad + nInput + i].ctypes.data, bufferD[bindingPad + nInput + i], bufferH[bindingPad + nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    for index in range(nProfile):
        cudart.cudaStreamSynchronize(stream)

    for index in range(nProfile):
        bindingPad = (nInput + nOutput) * index
        print("check result of context %d: %s" % (index, np.all(bufferH[bindingPad + 1] == bufferH[bindingPad + 0] + 1)))

    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    run([8, 8], 1)
    run([8, 8], 1)

    print("test finish!")
