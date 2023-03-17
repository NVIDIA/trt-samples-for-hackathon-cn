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

from cuda import cudart
import numpy as np
import tensorrt as trt

trtFile = "./model.plan"
timeCacheFile = "./model.cache"
nB, nC, nH, nW = 1, 1, 28, 28
np.random.seed(31193)
data = np.random.rand(nB, nC, nH, nW).astype(np.float32) * 2 - 1

class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, keepAll=True):
        super(MyAlgorithmSelector, self).__init__()
        self.keepAll = keepAll

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        if self.keepAll:  # keep all algorithms
            result = list((range(len(layerAlgorithmList))))
        else:  # select algorith by us
            # choose the algorithm spending longest time
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = list(np.argmax(timeList))

            # choose the algorithm spending smallest workspace
            #workspaceSizeList = [ algorithm.workspace_size for algorithm in layerAlgorithmList ]
            #result = list(np.argmin(workspaceSizeList))

            # choose one certain algorithm we have known (for building the same engine for many times)
            #if layerAlgorithmContext.name == "(Unnamed Layer* 0) [Convolution] + (Unnamed Layer* 1) [Activation]":
            #    # the number 2147483648 is from VERBOSE log, marking the certain algorithm
            #    result = [ index for index,algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 2147483648 ]

        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):  # report the tactic of the whole network
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

np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.algorithm_selector = MyAlgorithmSelector(True)  # set Algorithm Selector

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
