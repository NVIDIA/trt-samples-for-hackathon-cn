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

import numpy as np
import tensorrt as trt
from cuda import cudart

shape = [2, 3, 4, 5]
nContext = 2  # count of context
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profileList = [builder.create_optimization_profile() for _ in range(nContext)]
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1, -1])
layer = network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)
network.mark_output(layer.get_output(0))

for profile in profileList:
    profile.set_shape(inputT0.name, shape, shape, [k * nContext for k in shape])  # "* nContext" is just for this example, not required in real use case
    profile.set_shape(inputT1.name, shape, shape, [k * nContext for k in shape])
    config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_bindings
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = nIO - nInput
nIO, nInput, nOutput = nIO // nContext, nInput // nContext, nOutput // nContext

streamList = [cudart.cudaStreamCreate()[1] for _ in range(nContext)]
contextList = [engine.create_execution_context() for index in range(nContext)]

bufferH = []  # a list of buffers for all Context (all OptimizationProfile)
for index in range(nContext):
    stream = streamList[index]
    context = contextList[index]
    context.set_optimization_profile_async(index, stream)
    bindingPad = nIO * index  # skip bindings of previous OptimizationProfile occupied
    inputShape = [k * (index + 1) for k in shape]  # we use different shape for various context in this example, not required in real use case
    context.set_binding_shape(bindingPad + 0, inputShape)
    context.set_binding_shape(bindingPad + 1, inputShape)
    print("Context%d binding all? %s" % (index, "Yes" if context.all_binding_shapes_specified else "No"))
    for i in range(nIO):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))
    for i in range(nInput):
        bufferH.append(np.arange(np.prod(inputShape)).astype(np.float32).reshape(inputShape))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(bindingPad + nInput + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + nInput + i))))

bufferD = []
for i in range(len(bufferH)):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for index in range(nContext):
    print("Use Context %d" % index)
    stream = streamList[index]
    context = contextList[index]
    context.set_optimization_profile_async(index, stream)
    bindingPad = nIO * index
    inputShape = [k * (index + 1) for k in shape]
    context.set_binding_shape(bindingPad + 0, inputShape)
    context.set_binding_shape(bindingPad + 1, inputShape)
    for i in range(nIO * nContext):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    bufferList = [int(0) for b in bufferD[:bindingPad]] + [int(b) for b in bufferD[bindingPad:(bindingPad + nInput + nOutput)]] + [int(0) for b in bufferD[(bindingPad + nInput + nOutput):]]
    # divide the buffers into three parts, and fill int(0) for the parts beside the buffer of this context uses

    context.execute_async_v2(bufferList, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[bindingPad + nInput + i].ctypes.data, bufferD[bindingPad + nInput + i], bufferH[bindingPad + nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

for index in range(nContext):
    cudart.cudaStreamSynchronize(stream)

for index in range(nContext):
    bindingPad = nIO * index
    print("check result of context %d: %s" % (index, np.all(bufferH[bindingPad + 2] == bufferH[bindingPad + 0] + bufferH[bindingPad + 1])))

for stream in streamList:
    cudart.cudaStreamDestroy(stream)

for b in bufferD:
    cudart.cudaFree(b)
