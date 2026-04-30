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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast, check_api_coverage

@case_mark
def case_simple():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_padding_nd(tensor, [1, 2], [3, 4])
    # Input: T[shape0]
    # Output: T[shape1] where shape1[i] = shape0[i] for dims before last two, shape1[i] = shape0[i] + pre + post for last two
    # Data Type: T in [int8, int32, float16, float32]
    # Shape: len(shape0) >= 4; only the last two dimensions are padded or cropped
    layer.pre_padding_nd = [1, 2]  # Reset later, positive pads with zeros, negative trims
    layer.post_padding_nd = [3, 4]  # Reset later, positive pads with zeros, negative trims

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_crop():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_padding_nd(tensor, [-1, 0], [0, -2])

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Pad 0 around the input tensor
    case_simple()
    # Use pad layer to crop the input tensor
    case_crop()

    print("Finish")
