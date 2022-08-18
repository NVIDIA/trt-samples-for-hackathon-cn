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

import os
import numpy as np
from cuda import cudart
import tensorrt as trt

trtFile = "./model.plan"
timeCacheFile = "./model.cache"
nB, nC, nH, nW = 1, 1, 28, 28
np.random.seed(97)
data = np.random.rand(nB, nC, nH, nW).astype(np.float32) * 2 - 1

class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, keepAll=True):
        super(MyAlgorithmSelector, self).__init__()
        self.keepAll = keepAll

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        if self.keepAll:  # 保留全部选择，不做筛选
            result = list((range(len(layerAlgorithmList))))
        else:  # 手工筛选算法
            # 选择计算时间最长的算法
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = list(np.argmax(timeList))

            # 选择 workspace 最小的算法
            #workspaceSizeList = [ algorithm.workspace_size for algorithm in layerAlgorithmList ]
            #result = list(np.argmin(workspaceSizeList))

            # 让特定层选择特定算法（用于多次构建一模一样的引擎）
            #if layerAlgorithmContext.name == "(Unnamed Layer* 0) [Convolution] + (Unnamed Layer* 1) [Activation]":
            #    # 最后的数字来自 VERBOSE 日志中信息，代表该层的一种实现
            #    result = [ index for index,algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 2147483648 ]

        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):  # 报告整个网络优化后的  tactic
        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]

            print("Layer%4d:%s" % (i, context.name))
            nInput = context.num_inputs
            nOutput = context.num_outputs
            for j in range(nInput):
                ioInfo = algorithm.get_algorithm_io_info(j)
                print("    Input [%2d]:%s,%s,%s,%s" % (j, context.get_shape(j), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            for j in range(nOutput):
                ioInfo = algorithm.get_algorithm_io_info(j + nInput)
                print("    Output[%2d]:%s,%s,%s,%s" % (j, context.get_shape(j + nInput), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            print("    algorithm:[implementation:%d,tactic:%d,timing:%fms,workspace:%dMB]"% \
                  (algorithm.algorithm_variant.implementation,
                   algorithm.algorithm_variant.tactic,
                   algorithm.timing_msec,
                   algorithm.workspace_size))

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
config.algorithm_selector = MyAlgorithmSelector(True)  # 设置算法选择器

inputTensor = network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
profile.set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)

w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(32, 1, 1).astype(np.float32))
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(64, 1, 1).astype(np.float32))
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
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
if engineString == None:
    print("Failed building serialized engine!")
    exit()
print("Succeeded building serialized engine!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
if engine == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")

context = engine.create_execution_context()
context.set_binding_shape(0, [nB, nC, nH, nW])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput + nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

bufferH = []
bufferH.append(np.ascontiguousarray(data.reshape(-1)))
for i in range(nInput, nInput + nOutput):
    bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
bufferD = []
for i in range(nInput + nOutput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput + nOutput):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput + nOutput):
    print(engine.get_binding_name(i))

for b in bufferD:
    cudart.cudaFree(b)
