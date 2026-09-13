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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast

@case_mark
def case_simple():
    n_b, n_c, n_h, n_w = 1, 3, 4, 7  # batch, RNN batch size, sequence length, embedding width
    n_hidden = 5  # Hidden state width
    data = {"tensor": np.ones([n_b, n_c, n_h, n_w], dtype=np.float32)}
    weight_fx = np.ascontiguousarray(np.ones((n_w, n_hidden), dtype=np.float32))  # Forward weight matrix (X -> H)
    weight_fh = np.ascontiguousarray(np.ones((n_hidden, n_hidden), dtype=np.float32))  # Forward weight matrix (H -> H)
    weight_bx = np.ascontiguousarray(np.ones((n_w, n_hidden), dtype=np.float32))  # Backward weight matrix (X -> H)
    weight_bh = np.ascontiguousarray(np.ones((n_hidden, n_hidden), dtype=np.float32))  # Backward weight matrix (H -> H)
    bias_fx = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Forward bias (X -> H)
    bias_fh = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Forward bias (H -> H)
    bias_bx = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Backward bias (X -> H)
    bias_bh = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Backward bias (H -> H)

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    # Deprecated layer: IRNNv2Layer (add_rnn_v2) is removed since TensorRT 10; use ILoop structure instead.
    layer = tw.network.add_rnn_v2(tensor, 1, n_hidden, n_h, trt.RNNOperation.RELU)
    layer.direction = trt.RNNDirection.BIDIRECTION  # RNN direction, default: trt.RNNDirection.UNIDIRECTION
    layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(weight_fx))
    layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(weight_fh))
    layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(bias_fx))
    layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(bias_fh))
    layer.set_weights_for_gate(1, trt.RNNGateType.INPUT, True, trt.Weights(weight_bx))  # Backward pass is layer index 1
    layer.set_weights_for_gate(1, trt.RNNGateType.INPUT, False, trt.Weights(weight_bh))
    layer.set_bias_for_gate(1, trt.RNNGateType.INPUT, True, trt.Weights(bias_bx))
    layer.set_bias_for_gate(1, trt.RNNGateType.INPUT, False, trt.Weights(bias_bh))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A case of a bidirectional RNNv2 layer
    case_simple()

    print("Finish")
