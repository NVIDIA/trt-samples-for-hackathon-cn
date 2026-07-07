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
    data = {"tensor": np.full([3, 4, 5], 2, dtype=np.float32), "tensor1": np.full([3, 4, 5], 3, dtype=np.float32)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)
    # Input: T1[shape0], T1[shape1]
    # Output T2[shape2]
    # Data type:
    # | Operation |                       T1                       |  T2   |
    # | :-------: | :--------------------------------------------: | :---: |
    # |    SUM    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |   PROD    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |    MAX    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |    MIN    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |    SUB    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |    DIV    | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |   POWER   | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # | FLOOR_DIV | int8, int32, int64, float16, float32, bfloat16 |  T1   |
    # |    AND    |                      bool                      |  T1   |
    # |    OR     |                      bool                      |  T1   |
    # |    XOR    |                      bool                      |  T1   |
    # |   EQUAL   |    int32, int64, float16, float32, bfloat16    | bool  |
    # |  GREATER  |    int32, int64, float16, float32, bfloat16    | bool  |
    # |   LESS    |    int32, int64, float16, float32, bfloat16    | bool  |
    # Shape: shape0 == shape1 or broadcast when:
    # + The ranks of the two input tensors are same: len(shape0) == len(shape1).
    # + For each dimension of the two input tensors, either the lengths of this dimension are same, or at least one tensor has length of 1 at this dimension.

    layer.op = trt.ElementWiseOperation.SUM  # [Optional] Default: set at construction, trt.ElementWiseOperation.SUM in this case; Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_broadcast():
    n_c, n_h, n_w = 3, 4, 5
    data = {"tensor": np.full([n_c, 1, n_w], 1, dtype=np.float32), "tensor1": np.full([n_c, n_h, 1], 2, dtype=np.float32)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of compute elementewise addition
    case_simple()
    # Broadcast the elements while elementwise operation
    case_broadcast()

    print_enumerated_members(trt.ElementWiseOperation)

    print("Finish")
