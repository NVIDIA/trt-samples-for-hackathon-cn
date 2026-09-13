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
    n_b, n_c, n_h, n_w = 1, 3, 4, 5
    data = {"tensor": np.arange(n_b * n_c * n_h * n_w, dtype=np.float32).reshape(n_b, n_c, n_h, n_w)}
    factor_shape = data["tensor"].transpose(0, 1, 3, 2).shape  # (n_b, n_c, n_w, n_h)
    factor = np.ascontiguousarray(np.ones(factor_shape, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_constant = tw.network.add_constant(factor_shape, trt.Weights(factor))
    # Deprecated layer: add_matrix_multiply_deprecated is superseded by add_matrix_multiply (with trt.MatrixOperation).
    layer = tw.network.add_matrix_multiply_deprecated(tensor, True, layer_constant.get_output(0), True)
    # Input: A[shape0], B[shape1], batched matrix multiplication over the last two dimensions
    # Output: T[shape2]
    layer.transpose0 = False  # [Optional] Whether to transpose the first operand, default: set by constructor
    layer.transpose1 = False  # [Optional] Whether to transpose the second operand, default: set by constructor

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using the deprecated matrix-multiply layer
    case_simple()

    print("Finish")
