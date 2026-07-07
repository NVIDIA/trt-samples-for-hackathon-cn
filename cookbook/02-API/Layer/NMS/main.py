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
from tensorrt_cookbook import TRTWrapperDDS, case_mark, datatype_cast, print_enumerated_members, check_api_coverage

@case_mark
def case_simple():
    data = {"tensor": np.random.rand(60).astype(np.float32).reshape(5, 3, 4), "tensor1": np.random.rand(150).astype(np.float32).reshape(5, 3, 10)}

    tw = TRTWrapperDDS()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)

    layer_max_output = tw.network.add_constant([], np.int32(20).reshape(-1))
    layer = tw.network.add_nms(tensor0, tensor1, layer_max_output.get_output(0), trt.DataType.INT64)
    # Input: boxes: T1[nB, nInBox, nClass, 4] or T1[nB, nInBox, 4], scores: T1[nB, nInBox, nClass],
    #        max_output_boxes_per_class: int32[], iou_threshold (optional): float32[] in [0, 1] default 0, score_threshold (optional): float32[] default 0
    # Output: selected_indices: T2[nOutBox, 3] (rows are (batchIndex, classIndex, boxIndex)), num_output_boxes: int32[]
    # Data Type: T1 (boxes, scores) in [float16, float32, bfloat16], T2 (selected_indices) in [int32, int64]
    # Shape: np.prod(shape) <= 2**31 - 1 for boxes, scores and selected_indices each
    layer.indices_type = trt.DataType.INT64  # Reset later
    layer.topk_box_limit = 100  # [Optional] Default: 5000 (2000 on SM 5.3/6.2 devices); must be <= device-specific maximum
    layer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES  # [Optional] Default: trt.BoundingBoxFormat.CORNER_PAIRS

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_deprecated():
    data = {"tensor": np.random.rand(60).astype(np.float32).reshape(5, 3, 4), "tensor1": np.random.rand(150).astype(np.float32).reshape(5, 3, 10)}

    tw = TRTWrapperDDS()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)

    layer_max_output = tw.network.add_constant([], np.int32(20).reshape(-1))
    layer = tw.network.add_nms(tensor0, tensor1, layer_max_output.get_output(0))  # 3 parameters rather than 4
    layer.topk_box_limit = 100  # [Optional] Default: 5000 (2000 on SM 5.3/6.2 devices); must be <= device-specific maximum
    layer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES  # [Optional] Default: trt.BoundingBoxFormat.CORNER_PAIRS
    layer.indices_type = trt.DataType.INT64  # [Optional] Default: trt.DataType.INT32

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using NMS layer
    case_simple()
    # The same as case_simple but using deprecated API
    case_deprecated()

    print_enumerated_members(trt.BoundingBoxFormat)

    print("Finish")
