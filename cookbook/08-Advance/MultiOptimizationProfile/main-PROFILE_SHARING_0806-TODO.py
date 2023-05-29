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

nB, nC, nH, nW = 2, 1, 28, 28
shapeSmall = [nB, nC, nH, nW]
nProfile = 2  # count of OptimizationProfile we want to use
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profileList = [builder.create_optimization_profile() for index in range(nProfile)]
config = builder.create_builder_config()

config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

inputTensor = network.add_input("inputT0", trt.float32, [-1, nC, -1, -1])
profileList[0].set_shape(inputTensor.name, [1, nC, nH, nW], [2, nC, nH, nW], [4, nC, nH, nW])
config.add_optimization_profile(profileList[0])
profileList[1].set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH * 2, nW * 2], [nB * 2, nC, nH * 4, nW * 4])
config.add_optimization_profile(profileList[1])

w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(32).astype(np.float32))
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(64).astype(np.float32))
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.first_transpose = (0, 2, 3, 1)
_6.reshape_dims = (-1, 64 * 7 * 7)

w = np.ascontiguousarray(np.random.rand(64 * 7 * 7, 1024).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(1, 1024).astype(np.float32))
_7 = network.add_constant(w.shape, trt.Weights(w))
_8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
_9 = network.add_constant(b.shape, trt.Weights(b))
_10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
_11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

w = np.ascontiguousarray(np.random.rand(1024, 10).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
_12 = network.add_constant(w.shape, trt.Weights(w))
_13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
_14 = network.add_constant(b.shape, trt.Weights(b))
_15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

_16 = network.add_softmax(_15.get_output(0))
_16.axes = 1 << 1

_17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

network.mark_output(_17.get_output(1))

engineString = builder.build_serialized_network(network, config)
with open("model.plan", "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

cudaStreamList = [int(cudart.cudaStreamCreate()[1]) for i in range(nProfile)]
context = engine.create_execution_context()

bufferH = []  # use respective buffers for different Optimization Profile
for index in range(nProfile):
    context.set_optimization_profile_async(index, cudaStreamList[index])
    shape = [2, nC, 56, 56]
    context.set_input_shape(lTensorName[0], shape)
    for i in range(nInput):
        bufferH.append(np.arange(np.prod(shape)).astype(np.float32).reshape(shape))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(len(bufferH)):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for index in range(nProfile):
    print("Use Profile %d" % index)
    context.set_optimization_profile_async(index, cudaStreamList[index])  # set shape again after changing the optimization profile
    bindingPad = nIO * index
    shape = [2, nC, 56, 56]
    context.set_input_shape(lTensorName[0], shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, cudaStreamList[index])

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[bindingPad + i]))

    context.execute_async_v3(cudaStreamList[index])

    for i in range(nInput, nIO):
        cudart.cudaMemcpyAsync(bufferH[bindingPad + i].ctypes.data, bufferD[bindingPad + i], bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, cudaStreamList[index])

    cudart.cudaStreamSynchronize(cudaStreamList[index])

for index in range(nProfile):
    bindingPad = nIO * index
    print(bufferH[bindingPad + 1])

for b in bufferD:
    cudart.cudaFree(b)
