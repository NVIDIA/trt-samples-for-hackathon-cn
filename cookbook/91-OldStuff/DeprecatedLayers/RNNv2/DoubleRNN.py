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
    weight_x = np.ascontiguousarray(np.ones((n_hidden, n_w), dtype=np.float32))  # Weight matrix (X -> H)
    weight_h = np.ascontiguousarray(np.ones((n_hidden, n_hidden), dtype=np.float32))  # Weight matrix (H -> H)
    bias_x = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Bias (X -> H)
    bias_h = np.ascontiguousarray(np.zeros(n_hidden, dtype=np.float32))  # Bias (H -> H)

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    # Deprecated layer: IRNNv2Layer (add_rnn_v2) is removed since TensorRT 10; use ILoop structure instead.
    layer = tw.network.add_rnn_v2(tensor, 2, n_hidden, n_h, trt.RNNOperation.RELU)  # 2-layer ReLU RNN
    layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(weight_x))
    layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(weight_h))
    layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, trt.Weights(bias_x))
    layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, trt.Weights(bias_h))
    # Weights of the second layer, note the X transform now matches the hidden width
    layer.set_weights_for_gate(1, trt.RNNGateType.INPUT, True, trt.Weights(weight_h))
    layer.set_weights_for_gate(1, trt.RNNGateType.INPUT, False, trt.Weights(weight_h))
    layer.set_bias_for_gate(1, trt.RNNGateType.INPUT, True, trt.Weights(bias_h))
    layer.set_bias_for_gate(1, trt.RNNGateType.INPUT, False, trt.Weights(bias_h))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A case of stacking two RNNv2 layers
    case_simple()

    print("Finish")
