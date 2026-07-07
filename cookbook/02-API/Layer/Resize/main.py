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
from tensorrt_cookbook import (TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_cast, print_enumerated_members, check_api_coverage)

@case_mark
def case_simple():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    # Input: input: T[shape0], shape (optional): T1[len(shape0)]
    # Output: T[shape1]
    # Data Type: T in [float16, float32, bfloat16], T1 in [int32, int64]
    # Shape: shape1[i] = shape (attribute or input1)[i] when set, or shape1[i] = floor(shape0[i] * scales[i]) when scales set
    layer.shape = shape_output  # [Optional] Default: same as input shape; use either `shape` or `scales`
    #layer.scales = np.array(shape_output) / np.array(shape_input)  # [Optional] Default: [], use either `shape` or `scales`

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    constant_layer = tw.network.add_constant([4], np.array(shape_output, dtype=np.int32))
    layer = tw.network.add_resize(tensor)
    layer.set_input(1, constant_layer.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_cubic_mode():
    shape_input = 1, 3, 4, 5
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.resize_mode = trt.InterpolationMode.CUBIC  # [Optional] Default: NEAREST, options: NEAREST, LINEAR, CUBIC
    layer.cubic_coeff = 0.5  # [Optional] Default: -0.75, coefficient for cubic interpolation

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_linear():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = [shape_input[0], shape_output[1], 1, 1]  # [Optional] Default: same as input shape, output dimensions
    layer.resize_mode = trt.InterpolationMode.LINEAR  # [Optional] Default: NEAREST, options: NEAREST, LINEAR, CUBIC
    layer.selector_for_single_pixel = trt.ResizeSelector.UPPER  # [Optional] Default: FORMULA, options: FORMULA, UPPER
    layer.nearest_rounding = trt.ResizeRoundMode.CEIL  # [Optional] Default: FLOOR, options: HALF_UP, HALF_DOWN, FLOOR, CEIL
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS  # [Optional] Default: ASYMMETRIC, options: ALIGN_CORNERS, ASYMMETRIC, HALF_PIXEL

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_nearest_round_modes():
    # Exercise trt.InterpolationMode.NEAREST together with every trt.ResizeRoundMode and the FORMULA selector.
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}
    # All rounding modes valid for NEAREST interpolation.
    round_mode_list = [
        trt.ResizeRoundMode.FLOOR,
        trt.ResizeRoundMode.CEIL,
        trt.ResizeRoundMode.HALF_DOWN,
        trt.ResizeRoundMode.HALF_UP,
    ]

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    output_list = []
    for round_mode in round_mode_list:
        layer = tw.network.add_resize(tensor)
        layer.shape = shape_output
        layer.resize_mode = trt.InterpolationMode.NEAREST  # Nearest-neighbor interpolation
        layer.nearest_rounding = round_mode  # Rounding rule used by NEAREST interpolation
        layer.selector_for_single_pixel = trt.ResizeSelector.FORMULA  # Coordinate selector for single-pixel dimensions
        output_list.append(layer.get_output(0))

    tw.build(output_list)
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {
        "tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input),
        "tensor1": np.array(shape_output, dtype=np.int32),
    }

    tw = TRTWrapperShapeInput()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    tw.profile.set_shape_input(tensor1.name, [1 for _ in shape_input], shape_output, shape_output)

    layer = tw.network.add_resize(tensor0)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_exclude_outside():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output  # [Optional] Default: same as input shape
    layer.exclude_outside = 1  # [Optional] Default: 0 (False), range: {0, 1}, whether to exclude grid points outside the input boundary during resampling

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Set output shape or scales to resize the input tensor
    case_simple()
    # Use output shape from an earlier layer
    case_set_input()
    # Resize with cubic mode and related parameters
    case_cubic_mode()
    # Resize with linear mode and related parameters
    case_linear()
    # Resize with nearest mode exercising all round modes and the FORMULA selector
    case_nearest_round_modes()
    # Set output shape from shape input tensor
    case_shape_input()
    # Exclude grid points outside the input boundary during resampling
    case_exclude_outside()

    print_enumerated_members(trt.InterpolationMode)
    print_enumerated_members(trt.ResizeSelector)
    print_enumerated_members(trt.ResizeRoundMode)
    print_enumerated_members(trt.ResizeCoordinateTransformation)

    print("Finish")
