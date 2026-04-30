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
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_axis = tw.network.add_constant(shape=(), weights=np.array([1], dtype=np.int32))
    layer = tw.network.add_cumulative(tensor, layer_axis.get_output(0), trt.CumulativeOperation.SUM, False, False)
    # Input: input: T1[a0, ..., aN] (N >= 1), axis: T2[] (0D build-time constant)
    # Outputs: output: T1[a0, ..., aN] (same shape as input)
    # Data type: T1 in [int32, int64, float16, float32, bfloat16], T2 in [int32, int64]
    # Shape: axis must be in range [-rank(input), rank(input)-1]; negative values count dimensions backward
    # Volume limits: N/A
    layer.op = trt.CumulativeOperation.SUM  # Reset later
    layer.exclusive = False  # Reset later
    layer.reverse = False  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using cumulative layer
    case_simple()

    print_enumerated_members(trt.CumulativeOperation)

    print("Finish")
