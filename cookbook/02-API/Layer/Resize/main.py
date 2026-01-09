# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_np_to_trt)

@case_mark
def case_simple():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output
    #layer.scales = np.array(shape_output) / np.array(shape_input)  # Use either `shape` or `scales` is enough

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
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
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.resize_mode = trt.InterpolationMode.CUBIC
    layer.cubic_coeff = 0.5

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_linear():
    shape_input = 1, 3, 4, 5
    shape_output = 2, 3, 6, 10
    data = {"tensor": np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = [shape_input[0], shape_output[1], 1, 1]
    layer.resize_mode = trt.InterpolationMode.LINEAR
    layer.selector_for_single_pixel = trt.ResizeSelector.UPPER
    layer.nearest_rounding = trt.ResizeRoundMode.CEIL
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

    tw.build([layer.get_output(0)])
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
    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tw.profile.set_shape_input(tensor1.name, [1 for _ in shape_input], shape_output, shape_output)
    tw.config.add_optimization_profile(tw.profile)

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
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output
    layer.exclude_outside = 1

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
    # Set output shape from shape input tensor
    case_shape_input()
    # Use exclude outside (?)
    case_exclude_outside()

    print("Finish")
