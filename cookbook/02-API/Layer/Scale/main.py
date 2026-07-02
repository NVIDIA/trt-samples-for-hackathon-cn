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
    shape = [1, 3, 3, 3]
    data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    scale = np.ascontiguousarray(np.array([0.5], dtype=np.float32))
    shift = np.ascontiguousarray(np.array([-7.0], dtype=np.float32))
    power = np.ascontiguousarray(np.array([1.0], dtype=np.float32))
    layer = tw.network.add_scale(tensor, trt.ScaleMode.UNIFORM, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))
    # Input: T[shape0] (int8, float16, float32, bfloat16)
    # Outputs: T[shape0]
    # Data type: T in [int8, float16, float32, bfloat16] and weights.dtype == T
    # Shape: len(shape0) >= 4
    layer.mode = trt.ScaleMode.UNIFORM  # Reset later
    layer.shift = trt.Weights(shift)  # Reset later
    layer.scale = trt.Weights(scale)  # Reset later
    layer.power = trt.Weights(power)  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_channel():
    shape = [1, 3, 3, 3]
    data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
    scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
    layer = tw.network.add_scale(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_element():
    shape = [1, 3, 3, 3]
    data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    shift = np.ascontiguousarray(np.full(shape[1:], -7.0, dtype=np.float32))
    scale = np.ascontiguousarray(np.full(shape[1:], 0.5, dtype=np.float32))
    power = np.ascontiguousarray(np.ones(shape[1:], dtype=np.float32))
    layer = tw.network.add_scale(tensor, trt.ScaleMode.ELEMENTWISE, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_scale_channel_axis():
    shape = [1, 3, 3, 3]
    data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
    scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
    layer = tw.network.add_scale_nd(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power), 0)
    layer.channel_axis = 1  # [Optional] Default: 0

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of UNIFORM scale (one coefficient for all elements)
    case_simple()
    # CHANNEL scale (one coefficient per channel)
    case_channel()
    # ELEMENTWISE scale (one coefficient per element)
    case_element()
    # CHANNEL scale with a custom channel axis using add_scale_nd
    case_scale_channel_axis()

    print_enumerated_members(trt.ScaleMode)

    print("Finish")
