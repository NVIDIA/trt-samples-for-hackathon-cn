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
    shape = [1, 3, 4, 5]
    data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data = {
        "tensor": data0,
        "tensor1": -data0,
        "tensor2": (np.arange(np.prod(shape)) % 2).astype(bool).reshape(shape),
    }

    tw = TRTWrapperV1()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_cast(data["tensor2"].dtype, "trt"), data["tensor2"].shape)
    layer = tw.network.add_select(tensor2, tensor0, tensor1)
    # Input: condition (bool), thenInput (T), elseInput (T) — all must have the same rank; dimensions must match or be 1 (broadcast)
    # Outputs: output tensor of type T with shape determined by broadcasting inputs
    # Data type: T in {int32, int64, float16, float32, bfloat16, bool}
    # Shape: condition, thenInput and elseInput must have the same rank; for each dimension lengths must match or one must be 1 (broadcast); output shape follows broadcast rules

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Select layer
    case_simple()

    print("Finish")
