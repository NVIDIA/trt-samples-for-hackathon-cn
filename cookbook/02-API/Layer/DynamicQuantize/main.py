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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast
from packaging.version import Version

TRT_VERSION_GE_10_15 = Version(trt.__version__) >= Version("10.15")

@case_mark
def case_v1():
    data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

    tw = TRTWrapperV1(logger="verbose")
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP8, trt.DataType.FLOAT)
    layer.axis = 1  # [Optional] Reset axis later
    layer.block_size = 16  # [Optional] Reset block size later
    layer.to_type = trt.DataType.FP8  # [Optional] Reset target data type later
    layer.scale_type = trt.DataType.FLOAT  # [Optional] Reset scale data type later

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()
    return

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
    return

@case_mark
def case_v2():
    data = {"tensor": (np.arange(64, dtype=np.float32)).reshape(8, 8) / 32 - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_dynamic_quantize_v2(tensor, trt.Dims([4, 4]), trt.DataType.FP8, trt.DataType.FLOAT)
    layer.block_shape = [4, 4]  # [Optional] Reset block shape later
    layer.to_type = trt.DataType.FP8  # [Optional] Reset target data type later
    layer.scale_type = trt.DataType.FLOAT  # [Optional] Reset scale data type later

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()
    return

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
    return

if __name__ == "__main__":
    # A simple case of using dynamic-quantize layer
    case_v1()
    # A simple case of double quantization
    case_v1_double_quantization()

    if TRT_VERSION_GE_10_15:
        # v2
        case_v2()
        # v2 + double quantization (expected to fail)
        case_v2_double_quantization()

    print("Finish")
