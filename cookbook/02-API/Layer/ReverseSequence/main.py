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
    shape = [3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1) * 100 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1) * 10 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2]),
        "tensor1": np.array([3, 3, 2, 1], dtype=np.int32),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    layer = tw.network.add_reverse_sequence(tensor, tensor1)
    # Input: input: T[shape0], sequence_lens: T1[shape1]
    # Output: T[shape0]
    # Data Type: T in [float16, float32, bfloat16, int32, int64, int8, bool], T1 in [int32, int64]
    # Shape: len(shape1) == 1, shape1[0] == shape0[batch_axis]
    print(f"{layer.batch_axis = }, {layer.sequence_axis = }")  # batch_axis [Optional] Default: 1; sequence_axis [Optional] Default: 0

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_batch_prior():
    shape = [3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1) * 100 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1) * 10 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2]),
        "tensor1": np.array([3, 2, 1], dtype=np.int32),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    layer = tw.network.add_reverse_sequence(tensor, tensor1)
    layer.batch_axis = 0  # [Optional] Default: 1
    layer.sequence_axis = 1  # [Optional] Default: 0
    print(f"{layer.batch_axis = }, {layer.sequence_axis = }")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Reverse the input tensor of [SL, BS, H]
    case_simple()
    # Reverse the input tensor of [BS, SL, H]
    case_batch_prior()

    print("Finish")
