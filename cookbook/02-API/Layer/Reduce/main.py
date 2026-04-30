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
    data = {"tensor": np.ones([3, 4, 5], dtype=np.float32)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_reduce(tensor, trt.ReduceOperation.SUM, 1 << 1, False)
    # Input: T[shape0] with shape [a0, ..., an], n >= 1
    # Output: T[shape1]; keep_dims=True keeps reduced axes as size 1, keep_dims=False removes them (reduced rank)
    # Data type: T in [int8, int32, int64, float16, float32, bfloat16]
    # Shape: len(shape0) >= 1
    layer.op = trt.ReduceOperation.SUM  # Reset later
    layer.axes = 1 << 1  # Reset later, bitmask of axes to reduce (e.g. axes=6 reduces dims 1 and 2)
    layer.keep_dims = False  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Compute reduce sum on the second axis of input tensor
    case_simple()

    print_enumerated_members(trt.ReduceOperation)

    print("Finish")
