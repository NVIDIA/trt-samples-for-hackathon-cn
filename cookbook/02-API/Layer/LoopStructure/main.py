#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import (TRTWrapperShapeInput, TRTWrapperV1, case_mark)

shape = [1, 2, 3, 4]
data = {"inputT0": np.ones(shape, dtype=np.float32)}

@case_mark
def case_for():
    """
    # Behave like:
    layer_output_0 = tensor0
    layer_output_1 = np.zeros([t]+tensor0.shape)
    for i in range(t):
        layer_output_0 = layer_output_0 + layer_output_0
        layer_output_1[t] = layer_output_0
    return layer_output_0, layer_output_1
    """

    t = np.array([5], dtype=np.int32)  # number of iterations
    v = np.array([6], dtype=np.int32)  # number of outpu to keep, we usually use v == t

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)

    loop = tw.network.add_loop()

    layer_t = tw.network.add_constant((), t)
    loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

    layer_recurrence = loop.add_recurrence(tensor0)
    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output_0 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # Keep final output of the loop, argument index is ignored
    layer_output_1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)  # Keep all output during the loop
    # Keep output of iteration [1, t] if passing layer_body to loop.add_loop_output
    # Keep output of iteration [0, t-1] if passing layer_recurrence to loop.add_loop_output

    layer_v = tw.network.add_constant((), v)
    layer_output_1.set_input(1, layer_v.get_output(0))
    # Output shape on the iteration axis depends on v,
    # The output of first v ierations are kept if v <= t,
    # Or 0 padding is used for the part of v > t.

    tw.build([layer_output_0.get_output(0), layer_output_1.get_output(0)])  # Keep either or both of the output is OK
    tw.setup(data)
    tw.infer()

@case_mark
def case_for_set_input():
    """
    # Behave like:
    layer_output_0 = tensor0
    layer_output_1 = np.zeros([t]+tensor0.shape)
    for i in range(t):
        layer_output_0 = layer_output_0 + layer_output_0
        layer_output_1[t] = layer_output_0
    return layer_output_0, layer_output_1
    """

    data_v2 = {"inputT0": data["inputT0"], "inputT1": np.array([5], dtype=np.int32)}

    tw = TRTWrapperShapeInput()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.int32, ())  # set number of iterations at runtime
    tw.profile.set_shape_input(tensor1.name, [1], [6], [10])
    tw.config.add_optimization_profile(tw.profile)
    loop = tw.network.add_loop()

    loop.add_trip_limit(tensor1, trt.TripLimit.COUNT)

    layer_recurrence = loop.add_recurrence(tensor0)
    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output_0 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output_1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_output_1.set_input(1, tensor1)

    tw.build([layer_output_0.get_output(0), layer_output_1.get_output(0)])
    tw.setup(data_v2)
    tw.infer()

@case_mark
def case_weight_hile():
    """
    # Behave like:
    layer_output_0 = tensor0
    layer_output_1 = np.zeros([t]+inputT0.shape)
    weight_hile (layer_output_0.reshape(-1)[0] < 32)
        layer_output_1[t] = layer_output_0
        layer_output_0 += layer_output_0
    return layer_output_0, layer_output_1
    """

    threshold = np.array([32], dtype=np.float32)
    v = np.array([6], dtype=np.int32)  # number of outpu to keep

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)

    loop = tw.network.add_loop()
    layer_recurrence = loop.add_recurrence(tensor0)
    layer_threshold = tw.network.add_constant((), threshold)

    # Extract the scalar first element of `layer_recurrence`
    layer_1 = tw.network.add_shuffle(layer_recurrence.get_output(0))
    layer_1.reshape_dims = [-1]
    layer_2 = tw.network.add_slice(layer_1.get_output(0), [0], [1], [1])
    layer_3 = tw.network.add_shuffle(layer_2.get_output(0))
    layer_3.reshape_dims = []

    # Compare the element with threshold
    layer_4 = tw.network.add_elementwise(layer_threshold.get_output(0), layer_3.get_output(0), trt.ElementWiseOperation.SUB)
    layer_5 = tw.network.add_activation(layer_4.get_output(0), trt.ActivationType.RELU)
    layer_6 = tw.network.add_identity(layer_5.get_output(0))
    layer_6.set_output_type(0, trt.bool)
    layer_6.get_output(0).dtype = trt.bool

    loop.add_trip_limit(layer_6.get_output(0), trt.TripLimit.WHILE)

    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output_0 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output_1 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_v = tw.network.add_constant((), v)
    layer_output_1.set_input(1, layer_v.get_output(0))

    tw.build([layer_output_0.get_output(0), layer_output_1.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_iterator():
    """
    # Behave like:
    layer_output_0 = tensor0
    layer_output_1 = np.zeros([t]+tensor0.shape)
    for i in range(t):
        layer_output_0 = layer_output_0 + layer_output_0
        layer_output_1[t] = layer_output_0
    return layer_output_0, layer_output_1
    """
    _, nC, nH, nW = shape

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)

    loop = tw.network.add_loop()
    iterator = loop.add_iterator(tensor0, 1, False)  # Build a iterator with tensor, axis and weight_hether to reverse
    iterator.axis = 1  # Reset axis after constructor
    print(iterator.reverse)  # Read only sttribution

    layer_t = tw.network.add_constant((), np.array([nC], dtype=np.int32))
    loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

    layer_initial = tw.network.add_constant([1, nH, nW], np.ones(nH * nW, dtype=np.float32))
    rLayer = loop.add_recurrence(layer_initial.get_output(0))

    layer_body = tw.network.add_elementwise(rLayer.get_output(0), iterator.get_output(0), trt.ElementWiseOperation.SUM)
    rLayer.set_input(1, layer_body.get_output(0))

    layer_output_0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output_1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_output_1.set_input(1, layer_t.get_output(0))

    tw.build([layer_output_0.get_output(0), layer_output_1.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_unidirectional_lstm():
    nBS, nISL, nIH, nH = 3, 4, 7, 5  # batch_size, input_sequence_length, input_hidden_size, hidden_size
    x = np.ones([nBS, nISL, nIH], dtype=np.float32)
    h0 = np.ones([nBS, nH], dtype=np.float32)  # initial hidden state
    c0 = np.zeros([nBS, nH], dtype=np.float32)  # initial cell state
    data_v2 = {"x": x, "h0": h0, "c0": c0}

    weight_x = np.ones((nH, nIH), dtype=np.float32)  # weight of X->H, we use the same weight for each gate in this example
    weight_h = np.ones((nH, nH), dtype=np.float32)  # weight of H->H
    bias_x = np.zeros(nH, dtype=np.float32)  # bias of X->H
    bias_h = np.zeros(nH, dtype=np.float32)  # bias of H->H

    tw = TRTWrapperV1()
    input_x = tw.network.add_input("x", trt.float32, (-1, -1, nIH))
    input_h0 = tw.network.add_input("h0", trt.float32, (-1, nH))
    input_c0 = tw.network.add_input("c0", trt.float32, (-1, nH))
    tw.profile.set_shape(input_x.name, [1, 1, nIH], [nBS, nISL, nIH], [nBS, nISL * 2, nIH])
    tw.profile.set_shape(input_h0.name, [1, nH], [nBS, nH], [nBS, nH])
    tw.profile.set_shape(input_c0.name, [1, nH], [nBS, nH], [nBS, nH])
    tw.config.add_optimization_profile(tw.profile)

    def gate(tensor_x, weight_x, tensor_h, weight_h, bias, b_sigmoid):
        layer_h0 = tw.network.add_matrix_multiply(tensor_x, trt.MatrixOperation.NONE, weight_x, trt.MatrixOperation.NONE)
        layer_h1 = tw.network.add_matrix_multiply(tensor_h, trt.MatrixOperation.NONE, weight_h, trt.MatrixOperation.NONE)
        layer_h2 = tw.network.add_elementwise(layer_h0.get_output(0), layer_h1.get_output(0), trt.ElementWiseOperation.SUM)
        layer_h3 = tw.network.add_elementwise(layer_h2.get_output(0), bias, trt.ElementWiseOperation.SUM)
        layer_h4 = tw.network.add_activation(layer_h3.get_output(0), trt.ActivationType.SIGMOID if b_sigmoid else trt.ActivationType.TANH)
        return layer_h4

    loop = tw.network.add_loop()

    layer_t0 = tw.network.add_shape(input_x)
    layer_t1 = tw.network.add_slice(layer_t0.get_output(0), [1], [1], [1])
    layer_t2 = tw.network.add_shuffle(layer_t1.get_output(0))
    layer_t2.reshape_dims = ()
    layer_t3 = tw.network.add_cast(layer_t2.get_output(0), trt.DataType.INT32)
    loop.add_trip_limit(layer_t3.get_output(0), trt.TripLimit.COUNT)

    iterator = loop.add_iterator(input_x, 1, False)  # Get a slice [nBS, nIH] from input_x in each iteration
    tensor_x = iterator.get_output(0)
    layer_hidden_h = loop.add_recurrence(input_h0)  # initial hidden state and cell state. There are multiple loop variables in a loop.
    layer_hidden_c = loop.add_recurrence(input_c0)

    layer_weight_x = tw.network.add_constant([nIH, nH], trt.Weights(np.ascontiguousarray(weight_x.transpose())))
    layer_weight_h = tw.network.add_constant([nH, nH], trt.Weights(np.ascontiguousarray(weight_h.transpose())))
    layer_bias = tw.network.add_constant([1, nH], trt.Weights(np.ascontiguousarray(bias_x + bias_h)))

    weight_x = layer_weight_x.get_output(0)
    tensor_h = layer_hidden_h.get_output(0)
    weight_h = layer_weight_h.get_output(0)
    bias = layer_bias.get_output(0)
    gate_i = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
    gate_f = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
    gate_c = gate(tensor_x, weight_x, tensor_h, weight_h, bias, False)
    gate_o = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)

    layer_body1 = tw.network.add_elementwise(gate_f.get_output(0), layer_hidden_c.get_output(0), trt.ElementWiseOperation.PROD)
    layer_body2 = tw.network.add_elementwise(gate_i.get_output(0), gate_c.get_output(0), trt.ElementWiseOperation.PROD)
    layer_hidden_c1 = tw.network.add_elementwise(layer_body1.get_output(0), layer_body2.get_output(0), trt.ElementWiseOperation.SUM)
    layer_body3 = tw.network.add_activation(layer_hidden_c1.get_output(0), trt.ActivationType.TANH)
    layer_hidden_h1 = tw.network.add_elementwise(gate_o.get_output(0), layer_body3.get_output(0), trt.ElementWiseOperation.PROD)

    layer_hidden_h.set_input(1, layer_hidden_h1.get_output(0))
    layer_hidden_c.set_input(1, layer_hidden_c1.get_output(0))

    layer_output_0 = loop.add_loop_output(layer_hidden_h.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # shape [nBS,nHiddenSize]
    layer_output_0.get_output(0).name = "y"
    layer_output_1 = loop.add_loop_output(layer_hidden_h1.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # shape [nBS,nISL,nHiddenSize]
    layer_output_1.get_output(0).name = "h1"
    layer_output_1.set_input(1, layer_t2.get_output(0))
    layer_output_2 = loop.add_loop_output(layer_hidden_c.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # shape [nBS,nHiddenSize]
    layer_output_2.get_output(0).name = "c1"

    tw.build([layer_output_0.get_output(0), layer_output_1.get_output(0), layer_output_2.get_output(0)])

    tw.setup(data_v2)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Loop structure with For
    case_for()
    # A simple case of using Loop structure with For
    case_for_set_input()
    # A simple case of using Loop structure with While
    case_weight_hile()
    # USe iterator
    case_iterator()
    #
    case_unidirectional_lstm()

    print("Finish")
