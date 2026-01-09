# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_np_to_trt)

@case_mark
def case_for():
    """
    # Behave like:
    layer_output = tensor
    layer_output1 = np.zeros([t] + tensor.shape)
    for i in range(t):
        layer_output = layer_output + layer_output
        layer_output1[i] = layer_output
    return layer_output, layer_output1
    """
    data = {"tensor": np.ones([1, 2, 3, 4], dtype=np.float32)}
    t = np.array([5], dtype=np.int32)  # Number of iterations
    v = np.array([6], dtype=np.int32)  # Number of output to keep

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    loop = tw.network.add_loop()
    loop.name = "A cute Loop structure"

    layer_t = tw.network.add_constant((), t)
    loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

    layer_recurrence = loop.add_recurrence(tensor)
    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # Keep final output of the loop, argument index is ignored
    layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)  # Keep all output during the loop
    # Keep output of iteration [1, t] if passing layer_body to loop.add_loop_output
    # Keep output of iteration [0, t-1] if passing layer_recurrence to loop.add_loop_output

    layer_v = tw.network.add_constant((), v)
    layer_output1.set_input(1, layer_v.get_output(0))
    # Output shape on the iteration axis depends on v,
    # The output of first v ierations are kept if v <= t,
    # Or 0 padding is used for the part of v > t.

    tw.build([layer_output.get_output(0), layer_output1.get_output(0)])  # Keep either or both of the output is OK
    tw.setup(data)
    tw.infer()

@case_mark
def case_for_set_input():
    """
    # Behave like:
    layer_output = tensor
    layer_output1 = np.zeros([t] + tensor.shape)
    for i in range(t):
        layer_output = layer_output + layer_output
        layer_output1[i] = layer_output
    return layer_output, layer_output1
    """
    data = {"tensor": np.ones([1, 2, 3, 4], dtype=np.float32), "tensor1": np.array(5, dtype=np.int32)}

    tw = TRTWrapperShapeInput()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)  # Set number of iteration at runtime
    tw.profile.set_shape_input(tensor1.name, [1], [6], [10])
    tw.config.add_optimization_profile(tw.profile)
    loop = tw.network.add_loop()

    loop.add_trip_limit(tensor1, trt.TripLimit.COUNT)

    layer_recurrence = loop.add_recurrence(tensor)
    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_output1.set_input(1, tensor1)

    tw.build([layer_output.get_output(0), layer_output1.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_while():
    """
    # Behave like:
    layer_output = tensor
    layer_output1 = np.zeros([t] + tensor.shape)
    i = 0
    weight_hile (layer_output.reshape(-1)[0] < 32)
        layer_output1[i] = layer_output
        layer_output += layer_output
    return layer_output, layer_output1
    """
    data = {"tensor": np.ones([1, 2, 3, 4], dtype=np.float32)}
    threshold = np.array([32], dtype=np.float32)
    v = np.array([6], dtype=np.int32)  # Number of output to keep, we usually use v == t

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)

    loop = tw.network.add_loop()
    layer_recurrence = loop.add_recurrence(tensor)
    layer_threshold = tw.network.add_constant((), threshold)

    # Extract the scalar first element of `layer_recurrence`
    layer1 = tw.network.add_shuffle(layer_recurrence.get_output(0))
    layer1.reshape_dims = [-1]
    layer2 = tw.network.add_slice(layer1.get_output(0), [0], [1], [1])
    layer3 = tw.network.add_shuffle(layer2.get_output(0))
    layer3.reshape_dims = []

    # Compare the element with threshold
    layer4 = tw.network.add_elementwise(layer_threshold.get_output(0), layer3.get_output(0), trt.ElementWiseOperation.SUB)
    layer5 = tw.network.add_activation(layer4.get_output(0), trt.ActivationType.RELU)
    layer6 = tw.network.add_cast(layer5.get_output(0), trt.bool)

    loop.add_trip_limit(layer6.get_output(0), trt.TripLimit.WHILE)

    layer_body = tw.network.add_elementwise(layer_recurrence.get_output(0), layer_recurrence.get_output(0), trt.ElementWiseOperation.SUM)
    layer_recurrence.set_input(1, layer_body.get_output(0))

    layer_output = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output1 = loop.add_loop_output(layer_recurrence.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_v = tw.network.add_constant((), v)
    layer_output1.set_input(1, layer_v.get_output(0))

    tw.build([layer_output.get_output(0), layer_output1.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_iterator():
    """
    # Behave like:
    layer_output = tensor
    layer_output1 = np.zeros([t] + tensor.shape)
    for i in range(t):
        layer_output = layer_output + 1
        layer_output1[i] = layer_output
    return layer_output, layer_output1
    """
    data = {"tensor": np.ones([1, 2, 3, 4], dtype=np.float32)}
    _, n_c, n_h, n_w = data["tensor"].shape

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)

    loop = tw.network.add_loop()
    iterator = loop.add_iterator(tensor, 1, False)  # Build a iterator with tensor, axis and weight_hether to reverse
    iterator.axis = 1  # [Optional] Reset axis later
    print(f"{iterator.reverse=}")  # Read-only attribution

    layer_t = tw.network.add_constant((), np.array([n_c], dtype=np.int32))
    loop.add_trip_limit(layer_t.get_output(0), trt.TripLimit.COUNT)

    layer_initial = tw.network.add_constant([1, n_h, n_w], np.ones(n_h * n_w, dtype=np.float32))
    rLayer = loop.add_recurrence(layer_initial.get_output(0))

    layer_body = tw.network.add_elementwise(rLayer.get_output(0), iterator.get_output(0), trt.ElementWiseOperation.SUM)
    rLayer.set_input(1, layer_body.get_output(0))

    layer_output = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output1 = loop.add_loop_output(layer_body.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    layer_output1.set_input(1, layer_t.get_output(0))

    tw.build([layer_output.get_output(0), layer_output1.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_unidirectional_lstm():
    n_b, n_isl, n_ih, n_h = 3, 4, 7, 5  # batch_size, input_sequence_length, input_hidden_size, hidden_size
    x = np.ones([n_b, n_isl, n_ih], dtype=np.float32)
    h0 = np.ones([n_b, n_h], dtype=np.float32)  # Initial hidden state
    c0 = np.zeros([n_b, n_h], dtype=np.float32)  # Initial cell state
    data = {"x": x, "h0": h0, "c0": c0}

    weight_x = np.ones((n_h, n_ih), dtype=np.float32)  # Weight of X->H, we use the same weight for each gate in this example
    weight_h = np.ones((n_h, n_h), dtype=np.float32)  # Weight of H->H
    bias_x = np.zeros(n_h, dtype=np.float32)  # Bias of X->H
    bias_h = np.zeros(n_h, dtype=np.float32)  # Bias of H->H

    tw = TRTWrapperV1()
    input_x = tw.network.add_input("x", datatype_np_to_trt(data["x"].dtype), [-1, -1, n_ih])
    input_h0 = tw.network.add_input("h0", datatype_np_to_trt(data["h0"].dtype), [-1, n_h])
    input_c0 = tw.network.add_input("c0", datatype_np_to_trt(data["c0"].dtype), [-1, n_h])
    tw.profile.set_shape(input_x.name, [1, 1, n_ih], [n_b, n_isl, n_ih], [n_b, n_isl * 2, n_ih])
    tw.profile.set_shape(input_h0.name, [1, n_h], [n_b, n_h], [n_b, n_h])
    tw.profile.set_shape(input_c0.name, [1, n_h], [n_b, n_h], [n_b, n_h])
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

    iterator = loop.add_iterator(input_x, 1, False)  # Get a slice [n_b, n_ih] from input_x in each iteration
    tensor_x = iterator.get_output(0)
    layer_hidden_h = loop.add_recurrence(input_h0)  # Initial hidden state and cell state. There are multiple loop variables in a loop.
    layer_hidden_c = loop.add_recurrence(input_c0)

    layer_weight_x = tw.network.add_constant([n_ih, n_h], trt.Weights(np.ascontiguousarray(weight_x.transpose())))
    layer_weight_h = tw.network.add_constant([n_h, n_h], trt.Weights(np.ascontiguousarray(weight_h.transpose())))
    layer_bias = tw.network.add_constant([1, n_h], trt.Weights(np.ascontiguousarray(bias_x + bias_h)))

    weight_x = layer_weight_x.get_output(0)
    tensor_h = layer_hidden_h.get_output(0)
    weight_h = layer_weight_h.get_output(0)
    bias = layer_bias.get_output(0)
    gate_i = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
    gate_f = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)
    gate_c = gate(tensor_x, weight_x, tensor_h, weight_h, bias, False)
    gate_o = gate(tensor_x, weight_x, tensor_h, weight_h, bias, True)

    layer_body = tw.network.add_elementwise(gate_f.get_output(0), layer_hidden_c.get_output(0), trt.ElementWiseOperation.PROD)
    layer_body1 = tw.network.add_elementwise(gate_i.get_output(0), gate_c.get_output(0), trt.ElementWiseOperation.PROD)
    layer_hidden_c1 = tw.network.add_elementwise(layer_body.get_output(0), layer_body1.get_output(0), trt.ElementWiseOperation.SUM)
    layer_body2 = tw.network.add_activation(layer_hidden_c1.get_output(0), trt.ActivationType.TANH)
    layer_hidden_h1 = tw.network.add_elementwise(gate_o.get_output(0), layer_body2.get_output(0), trt.ElementWiseOperation.PROD)

    layer_hidden_h.set_input(1, layer_hidden_h1.get_output(0))
    layer_hidden_c.set_input(1, layer_hidden_c1.get_output(0))

    layer_output = loop.add_loop_output(layer_hidden_h.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output.get_output(0).name = "y"
    layer_output1 = loop.add_loop_output(layer_hidden_h1.get_output(0), trt.LoopOutput.CONCATENATE, 1)
    layer_output1.get_output(0).name = "h1"
    layer_output1.set_input(1, layer_t2.get_output(0))
    layer_output2 = loop.add_loop_output(layer_hidden_c.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    layer_output2.get_output(0).name = "c1"

    tw.build([layer_output.get_output(0), layer_output1.get_output(0), layer_output2.get_output(0)])

    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # For loop
    case_for()
    # For loop with shape input tensor to decide iteration time
    case_for_set_input()
    # While loop
    case_while()
    # Use iterator in loop structure
    case_iterator()
    # Use loop to implement LSTM
    case_unidirectional_lstm()

    print("Finish")
