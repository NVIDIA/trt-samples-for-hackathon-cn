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
def case_v1():
    data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

    tw = TRTWrapperV1(logger="verbose")
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP8, trt.DataType.FLOAT)
    # Input: input: T1[shape0], double_quant_scale: T1[shape1]
    # Output: output: T2[shape0], scale: T3[shape2]
    # Data type: T1 in [float16, bfloat16, float32], T2 in [float4, float8], T3 in [float8, float32]
    # Shape: len(shape0) in [2, 3] for 1D block quantization, len(shape0) in [2, 3, 4] for 2D block quantization, len(shape1) == 0
    # len(shape2) == len(shape0), shape2[i] == shape0[i] for non-quantization axis, and shape2[i] == shape0[i] / block_shape[i] for quantization axis, block_shape[i] can be block_size

    layer.axis = 1  # [Optional] The axis sliced into blocks; must be last or second-to-last dimension
    layer.block_size = 16  # [Optional] Number of elements sharing a scale factor; valid values: 16 or 32
    layer.to_type = trt.DataType.FP8  # [Optional] Data type of quantized output; valid values: DataType.FP4, DataType.FP8
    layer.scale_type = trt.DataType.FLOAT  # [Optional] Data type of scale factor; valid values: DataType.FP8, DataType.FLOAT

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_v1_double_quantization():
    data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
    layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
    layer.set_input(1, double_quantization_layer.get_output(0))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_v2():
    data = {"tensor": (np.arange(64, dtype=np.float32)).reshape(8, 8) / 32 - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_dynamic_quantize_v2(tensor, trt.Dims([4, 4]), trt.DataType.FP8, trt.DataType.FLOAT)
    layer.block_shape = [4, 4]  # [Optional] Shape of the block to quantize; same rank as input, -1 matches input extent
    layer.to_type = trt.DataType.FP8  # [Optional] Data type of quantized output; valid values: DataType.FP4, DataType.FP8
    layer.scale_type = trt.DataType.FLOAT  # [Optional] Data type of scale factor; valid values: DataType.FP8, DataType.FLOAT

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_v2_double_quantization():
    data = {"tensor": (np.arange(64, dtype=np.float32)).reshape(8, 8) / 32 - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
    layer = tw.network.add_dynamic_quantize_v2(tensor, trt.Dims([4, 4]), trt.DataType.FP4, trt.DataType.FP8)
    layer.set_input(1, double_quantization_layer.get_output(0))

    try:
        tw.build([layer.get_output(0), layer.get_output(1)])
    except Exception:
        print("case_v2_double_quantization is expected to fail on current TensorRT")

if __name__ == "__main__":
    # A simple case of using dynamic-quantize layer
    case_v1()
    # A simple case of double quantization
    case_v1_double_quantization()
    # v2
    case_v2()
    # v2 + double quantization (expected to fail)
    case_v2_double_quantization()

    print_enumerated_members(trt.DataType)

    print("Finish")
