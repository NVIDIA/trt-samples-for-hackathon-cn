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
from tensorrt_cookbook import (TRTWrapperDDS, TRTWrapperV1, TRTWrapperV2, case_mark, datatype_cast, print_enumerated_members, check_api_coverage)

@case_mark
def case_simple():
    data = {"tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1, trt.DataType.INT64)
    # Input: input: T1[shape0]
    # Output: values: T1[shape1], indices: T2[shape1]
    # Data type: T1 in [int32, int64, float16, float32, bfloat16], T2 in [int32, int64]
    # Shape: shape1[i] == shape0[i] if i != log2(axes) else k
    # Entry with smaller index will be selected if they own the same value.
    # k <= d (axis dimension size) and k <= 3840
    # axes must be one of the last four dimensions (bitmask)

    layer.op = trt.TopKOperation.MAX  # Reset later
    layer.k = 2  # Reset later
    layer.axes = 1 << 1  # Reset later
    layer.indices_type = trt.DataType.INT64  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_deprecated():
    data = {"tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1)  # 3 parameters rather than 4
    layer.op = trt.TopKOperation.MAX  # Reset later
    layer.k = 2  # Reset later
    layer.axes = 1 << 1  # Reset later
    layer.indices_type = trt.DataType.INT64  # [Optional] Default: trt.DataType.INT32, options: trt.DataType.INT32, trt.DataType.INT64

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    data = {
        "tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5),
        "tensor1": np.array([2], dtype=np.int32),  # One more shape input tensor
    }

    tw = TRTWrapperV2()  # Use Data-Dependent-Shape and Shape-Input mode at the same time

    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), [])
    tw.profile.set_shape_input(tensor1.name, [1], [2], [3])

    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dds():
    data = {
        "tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5),
        "tensor1": np.array([3, -1], dtype=np.int32),  # tensor1 is a execution input tensor
    }

    tw = TRTWrapperDDS()  # Use Data-Dependent-Shape mode

    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), [-1 for _ in data["tensor1"].shape])
    tw.profile.set_shape(tensor1.name, [1 for _ in data["tensor1"].shape], data["tensor1"].shape, data["tensor1"].shape)

    layer1 = tw.network.add_reduce(tensor1, trt.ReduceOperation.SUM, 1 << 0, False)  # Compute K from earlier layer
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, layer1.get_output(0))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Get Top 2 from input tensor
    case_simple()
    # The same as case_simple but using deprecated API
    case_deprecated()
    # Use K from shape input tensor
    case_shape_input()
    # Use K from output of earlier layer
    case_dds()

    print_enumerated_members(trt.TopKOperation)
    print_enumerated_members(trt.ReduceOperation)

    print("Finish")
