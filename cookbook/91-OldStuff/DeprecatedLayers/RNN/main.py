# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

import numpy as np
import tensorrt as trt
from cuda import cudart
from tensorrt_cookbook import case_mark

@case_mark
def case_simple():
    # Deprecated layer: IRNNLayer (add_rnn) only works in implicit-batch mode and is removed since TensorRT 10; use ILoop structure instead.
    # It cannot be expressed with TRTWrapperV1 (explicit batch), so minimal manual builder / runtime boilerplate is kept here.
    n_b, n_c, n_h, n_w = 1, 3, 4, 7  # batch, RNN batch size, sequence length, embedding width
    n_hidden = 5  # Hidden state width
    data = np.ones(n_c * n_h * n_w, dtype=np.float32).reshape(n_c, n_h, n_w)
    weight = np.ascontiguousarray(np.ones((n_hidden, n_w + n_hidden), dtype=np.float32))  # Weight matrix, X and H concatenated together
    bias = np.ascontiguousarray(np.zeros(n_hidden * 2, dtype=np.float32))  # Bias, bX and bH concatenated together

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()  # implicit-batch mode is mandatory for add_rnn
    builder_config = builder.create_builder_config()
    tensor = network.add_input("inputT0", trt.float32, (n_c, n_h, n_w))

    layer_shuffle = network.add_shuffle(tensor)  # Shuffle into (n_h, n_c, n_w) first
    layer_shuffle.first_transpose = (1, 0, 2)
    fake_weight = trt.Weights(np.random.rand(n_hidden, n_w + n_hidden).astype(np.float32))
    fake_bias = trt.Weights(np.random.rand(n_hidden * 2).astype(np.float32))
    layer = network.add_rnn(layer_shuffle.get_output(0), 1, n_hidden, n_h, trt.RNNOperation.RELU, trt.RNNInputMode.LINEAR, trt.RNNDirection.UNIDIRECTION, fake_weight, fake_bias)
    layer.weights = trt.Weights(weight)  # Reset the RNN weights
    layer.bias = trt.Weights(bias)  # Reset the RNN bias

    network.mark_output(layer.get_output(0))
    network.mark_output(layer.get_output(1))
    engine = builder.build_engine(network, builder_config)
    context = engine.create_execution_context()
    n_input = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    n_output = engine.num_bindings - n_input

    buffer_h = [data]
    for i in range(n_output):
        buffer_h.append(np.empty(context.get_binding_shape(n_input + i), dtype=trt.nptype(engine.get_binding_dtype(n_input + i))))
    buffer_d = []
    for i in range(engine.num_bindings):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])

    for i in range(n_input):
        cudart.cudaMemcpy(buffer_d[i], np.ascontiguousarray(buffer_h[i].reshape(-1)).ctypes.data, buffer_h[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute(n_b, buffer_d)
    for i in range(n_output):
        cudart.cudaMemcpy(buffer_h[n_input + i].ctypes.data, buffer_d[n_input + i], buffer_h[n_input + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(n_input):
        print(f"Input {i}:", buffer_h[i].shape, "\n", buffer_h[i])
    for i in range(n_output):
        print(f"Output {i}:", buffer_h[n_input + i].shape, "\n", buffer_h[n_input + i])

    for buffer in buffer_d:
        cudart.cudaFree(buffer)

if __name__ == "__main__":
    # A simple case of using the deprecated implicit-batch RNN layer
    case_simple()

    print("Finish")
