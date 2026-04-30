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
    data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_unary(tensor, trt.UnaryOperation.ABS)
    # Input: T[shape0]
    # Outputs: T1[shape0]
    # Data type:
    #   T1 is bool if op in [ISINF, ISNAN] else T1 == T
    #   T is bool if op in [NOT]
    #   T in [int8,int32,int64,float16,float32,bfloat16] if op in [ABS,NEG,SIGN]
    #   T in [int8,float16,float32,bfloat16] if op in [SIN,COS,TAN,ASIN,ACOS,ATAN,SINH,COSH,ASINH,ACOSH,ATANH,EXP,LOG,SQRT,RECIP,CEIL,FLOOR,ERF,ROUND,ISINF,ISNAN]

    layer.op = trt.UnaryOperation.ABS  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Compute absolute values of the tensor
    case_simple()

    print_enumerated_members(trt.UnaryOperation)

    print("Finish")
