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

# For TensorRT < 8.5 with deprecated binding API

import numpy as np
import tensorrt as trt
from cuda import cudart

shapeSmall = [2, 3, 4, 5]
nProfile = 2  # count of OptimizationProfile we want to use
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profileList = [builder.create_optimization_profile() for index in range(nProfile)]
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1, -1])
for profile in profileList:
    profile.set_shape(inputT0.name, shapeSmall, shapeSmall, (np.array(shapeSmall) * 2).tolist())
    profile.set_shape(inputT1.name, shapeSmall, shapeSmall, (np.array(shapeSmall) * 2).tolist())
    config.add_optimization_profile(profile)

layer = network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)
network.mark_output(layer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_bindings  # NOTICE: this nIO is different from the result of engine.num_io_tensors
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
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
    context.set_binding_shape(bindingPad + 1, shape)
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
    context.set_binding_shape(bindingPad + 1, shape)
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
    print("check OptimizationProfile %d: %s" % (index, np.all(bufferH[bindingPad + 2] == bufferH[bindingPad + 0] + bufferH[bindingPad + 1])))

for b in bufferD:
    cudart.cudaFree(b)
