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
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_concatenation([tensor, tensor])
    # Input: list[T[shape_i]], limited up to 10000 tensors.
    # Outputs: T[shape1]
    # Data Type: T in [bool, int8, int32, int64, float16, float32, bfloat16]
    # Shape: len(shape_i.shape) == len(shape_j.shape) except for the axis dimension
    layer.axis = 2  # [Optional] Default: max(0, len(tensor.shape)-3)

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of concatenate two tensoers together
    case_simple()

    print("Finish")
