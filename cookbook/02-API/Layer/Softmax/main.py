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
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_softmax(tensor)
    # Input: T[shape0]
    # Output: T[shape0]
    # Data type: T in [float16, float32, bfloat16]
    # Shape: input and output share the same shape [a0, ..., an]
    layer.axes = 1 << 1  # [Optional] Default: 1 << max(0, Rank(tensor) - 3), bitmask of the single axis to normalize (only one axis allowed)

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Compute softmax
    case_simple()

    print("Finish")
