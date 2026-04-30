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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast, print_enumerated_members, check_api_coverage

@case_mark
def case_simple():
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_activation(tensor, trt.ActivationType.RELU)
    # Input: T[shape0]
    # Output: T[shape0]
    # Data Type: T in [float16, float32, bfloat16], and extra T in [int32, int64] for only RELU.
    layer.type = trt.ActivationType.RELU  # Reset later
    layer.alpha = -2  # [Optional] Parameter for LEAKY_RELU, ELU, SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH, THRESHOLDED_RELU, default: 0
    layer.beta = 2  # [Optional] Parameter for SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH, default: 0

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using ReLU activation layer
    case_simple()

    print_enumerated_members(trt.ActivationType)

    print("Finish")
