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
    layer = tw.network.add_shape(tensor)
    # Input: tensor of type T1 (bool, int4, int8, int32, int64, float8, float16, float32, bfloat16)
    # Outputs: tensor of type T2 (int64)
    # Data type: T1 supports bool/int4/int8/int32/int64/float8/float16/float32/bfloat16; T2 is int64
    # Shape: input shape [a0,...,an] where n>=0; output is a shape tensor with values [a0,...,an]; if n==0 output is empty tensor

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_mark_output_for_shapes():
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_shape(tensor)

    tw.network.mark_output_for_shapes(layer.get_output(0))
    tw.build()
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Get shape of a tensor
    case_simple()
    # Use `mark_output_for_shapes` to output a shape tensor
    #case_mark_output_for_shapes()  # TODO: not finish

    print("Finish")
