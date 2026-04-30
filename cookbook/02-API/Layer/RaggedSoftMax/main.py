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
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, datatype_cast

@case_mark
def case_simple():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.ones(np.prod(shape), dtype=np.float32).reshape(shape[1:]),
        "tensor1": np.tile(2 * np.arange(shape[2], dtype=np.int32), (shape[1], 1)).reshape(shape[1], shape[2], 1),
    }

    tw = TRTWrapperV1()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    layer = tw.network.add_ragged_softmax(tensor0, tensor1)
    # Input: input tensor of type T1; bounds tensor of type T2 describing sequence lengths
    # Outputs: output tensor of type T1
    # Data type: T1 is float16, float32, or bfloat16; T2 is int32
    # Shape: input and output are tensors with shape [a0,...,an] where n in (2, 3)

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using ragged softmax layer
    case_simple()

    print("Finish")
