#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

def get_mnist_network(config: trt.IBuilderConfig, network: trt.INetworkDefinition, profile: trt.IOptimizationProfile):

    shape = [-1, 1, 28, 28]

    tensor = network.add_input("inputT0", trt.float32, shape)
    profile.set_shape(tensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
    config.add_optimization_profile(profile)

    w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(32, 1, 1).astype(np.float32))
    layer = network.add_convolution_nd(tensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
    layer.name = "Convolution1"
    layer.padding_nd = [2, 2]
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation1"
    layer = network.add_pooling_nd(layer.get_output(0), trt.PoolingType.MAX, [2, 2])
    layer.name = "Pooling1"
    layer.stride_nd = [2, 2]

    w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(64, 1, 1).astype(np.float32))
    layer = network.add_convolution_nd(layer.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
    layer.name = "Convolution2"
    layer.padding_nd = [2, 2]
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation2"
    layer = network.add_pooling_nd(layer.get_output(0), trt.PoolingType.MAX, [2, 2])
    layer.name = "Pooling2"
    layer.stride_nd = [2, 2]

    layer = network.add_shuffle(layer.get_output(0))
    layer.name = "Shuffle"
    layer.reshape_dims = (-1, 64 * 7 * 7)

    w = np.ascontiguousarray(np.random.rand(64 * 7 * 7, 1024).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(1, 1024).astype(np.float32))
    constant_layer = network.add_constant(w.shape, trt.Weights(w))
    constant_layer.name = "MatrixMultiplication1Weight"
    layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    layer.name = "MatrixMultiplication1"
    constant_layer = network.add_constant(b.shape, trt.Weights(b))
    constant_layer.name = "ConstantBias1"
    layer = network.add_elementwise(layer.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.SUM)
    layer.name = "AddBias1"
    layer = network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
    layer.name = "Activation3"

    w = np.ascontiguousarray(np.random.rand(1024, 10).astype(np.float32))
    b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
    constant_layer = network.add_constant(w.shape, trt.Weights(w))
    constant_layer.name = "MatrixMultiplication2Weight"
    layer = network.add_matrix_multiply(layer.get_output(0), trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    layer.name = "MatrixMultiplication2"
    constant_layer = network.add_constant(b.shape, trt.Weights(b))
    constant_layer.name = "ConstantBias2"
    layer = network.add_elementwise(layer.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.SUM)
    layer.name = "AddBias2"
    layer = network.add_softmax(layer.get_output(0))
    layer.name = "Softmax"
    layer.axes = 1 << 1

    layer = network.add_topk(layer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
    layer.name = "TopK"
    return [layer.get_output(0)]

"""
def get_a_network(tw: TRTWrapperV1 = None, input_shape: list = [3, 4], n_layers: int = 10, n_hidden_size: int = 1024, n_profile: int = 1):

    tensor = tw.network.add_input("inputT0", trt.float32, [-1, input_shape[-1]])
    for i in range(n_profile):
        if i == 0:
            profile = tw.profile if i == 0 else tw.builder.create_optimization_profile()
        profile.set_shape(tensor.name, [1, input_shape[-1]], [input_shape[0] * (i + 1), input_shape[-1]], [input_shape[0] * (i + 1), input_shape[-1]])
        tw.config.add_optimization_profile(profile)

    for i in range(n_layers):
        n_input_size = input_shape[-1] if i == 0 else n_hidden_size
        w = np.ascontiguousarray(np.random.rand(n_input_size, n_hidden_size).astype(np.float32))
        b = np.ascontiguousarray(np.random.rand(1, n_hidden_size).astype(np.float32))
        w = tw.network.add_constant(w.shape, trt.Weights(w))
        b = tw.network.add_constant(b.shape, trt.Weights(b))
        layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, w.get_output(0), trt.MatrixOperation.NONE)
        tensor = layer.get_output(0)
        layer = tw.network.add_elementwise(tensor, b.get_output(0), trt.ElementWiseOperation.SUM)
        tensor = layer.get_output(0)
        layer = tw.network.add_activation(tensor, trt.ActivationType.RELU)
        tensor = layer.get_output(0)

    tensor.name = "outputT0"
    return [tensor]
"""
