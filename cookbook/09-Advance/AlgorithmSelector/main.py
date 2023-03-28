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

shape = [1, 1, 28, 28]
np.random.seed(31193)
data = np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

def getSizeString(xByte):
    if xByte < (1 << 10):
        return "%5.1f  B" % xByte
    if xByte < (1 << 20):
        return "%5.1fKiB" % (xByte / (1 << 10))
    if xByte < (1 << 30):
        return "%5.1fMiB" % (xByte / (1 << 20))
    return "%5.1fGiB" % (xByte / (1 << 30))

class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, iStrategy=0):
        super(MyAlgorithmSelector, self).__init__()
        self.iStrategy = iStrategy

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        # some bug in report_algorithms to make the algorithm.timing_msec and algorithm.workspace_size are always 0?
        # we print the alternative algorithms of each layer here
        nInput = layerAlgorithmContext.num_inputs
        nOutput = layerAlgorithmContext.num_outputs
        print("Layer %s,in=%d,out=%d" % (layerAlgorithmContext.name, nInput, nOutput))
        for i in range(nInput + nOutput):
            print("    %s    %2d: shape=%s" % ( \
                "Input " if i < nInput else "Output",
                i if i< nInput else i - nInput,
                layerAlgorithmContext.get_shape(i)))
        for i, algorithm in enumerate(layerAlgorithmList):
            print("    algorithm%3d:implementation[%10d], tactic[%20d], timing[%7.3fus], workspace[%s]" % ( \
                i,
                algorithm.algorithm_variant.implementation,
                algorithm.algorithm_variant.tactic,
                algorithm.timing_msec * 1000,
                getSizeString(algorithm.workspace_size)))

        if self.iStrategy == 0:  # choose the algorithm spending shortest time, the same as TensorRT
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmin(timeList)]

        elif self.iStrategy == 1:  # choose the algorithm spending longest time, just for fun getting a TensorRT engine with worst performance :)
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmax(timeList)]

        elif self.iStrategy == 2:  # choose the algorithm using smallest workspace
            workspaceSizeList = [algorithm.workspace_size for algorithm in layerAlgorithmList]
            result = [np.argmin(workspaceSizeList)]

        elif self.iStrategy == 3:  # choose one certain algorithm we have known
            # This strategy can be a workaround for building the exactly same engine for many times, but Timing Cache (refer to 09-Advance/TimingCache) is more recommended to do so.
            # The reason is that function select_algorithms is called after the performance test of all algorithms for a layer is finished (you can find algorithm.timing_msec > 0),
            # so it will not save the time of the test.
            # On the contrary, performance test of the algorithms will be skiped using Timing Cache (though performance test of Reformating can not be skiped),
            # so it surely saves a lot of time comparing with Algorithm Selector.
            if layerAlgorithmContext.name == "(Unnamed Layer* 0) [Convolution] + (Unnamed Layer* 1) [Activation]":
                # the number 2147483648 is from VERBOSE log, marking the certain algorithm
                result = [index for index, algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 2147483648]
            else:  # keep all algorithms for other layers
                result = list(range(len(layerAlgorithmList)))

        else:  # default behavior: keep all algorithms
            result = list(range(len(layerAlgorithmList)))

        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):  # report the tactic of the whole network
        print("[MyAlgorithmSelector::report_algorithms]")
        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]
            nInput = context.num_inputs
            nOutput = context.num_outputs
            print("Layer %s,in=%d,out=%d" % (context.name, nInput, nOutput))
            for i in range(nInput + nOutput):
                ioInfo = algorithm.get_algorithm_io_info(i)
                print("    %s    %2d: %s stride=%s, vectorized_dim=%d, components_per_element=%d, shape=%s" % ( \
                    "Input " if i < nInput else "Output",
                    i if i < nInput else  i - nInput,
                    ioInfo.dtype,
                    ioInfo.strides,
                    ioInfo.vectorized_dim,
                    ioInfo.components_per_element,
                    context.get_shape(i)))
            print("    algorithm   :implementation[%10d], tactic[%20d], timing[%7.3fus], workspace[%s]" % ( \
                algorithm.algorithm_variant.implementation,
                algorithm.algorithm_variant.tactic,
                algorithm.timing_msec * 1000,
                getSizeString(algorithm.workspace_size)))

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.algorithm_selector = MyAlgorithmSelector(1)  # assign Algorithm Selector to BuilderConfig, number here is the index of my customerized strategies to select algorithm
config.set_flag(trt.BuilderFlag.FP16)  # add FP16 to  get more alternative algorithms

inputTensor = network.add_input("inputT0", trt.float32, [-1] + shape[1:])
profile.set_shape(inputTensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
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
