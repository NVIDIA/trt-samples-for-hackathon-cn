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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

@case_mark
def case_simple():
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_identity(tensor)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_datatype_conversion():
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.FP16)  # Needed if using float16
    tw.config.set_flag(trt.BuilderFlag.BF16)  # Needed if using bfloat16
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    output_tensor_list = []
    for data_type in [trt.float16, trt.bfloat16, trt.int32, trt.int64, trt.uint8, trt.bool]:  # exclude trt.int8 and trt.int4
        # FP8 / FP4 is only supported from Plugin / Quantize / Constant / Concatenation / Shuffle layer
        layer = tw.network.add_cast(tensor, data_type)
        output_tensor_list.append(layer.get_output(0))

    tw.build(output_tensor_list)
    tw.setup(data)
    tw.infer()

@case_mark
def case_datatype_conversion_int8():
    data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)  # Needed if using int8
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    output_tensor_list = []
    for data_type in [trt.int8]:
        layer = tw.network.add_cast(tensor, data_type)
        layer.get_output(0).set_dynamic_range(0, 127)  # dynamic range or calibration needed for INT8
        output_tensor_list.append(layer.get_output(0))

    tw.build(output_tensor_list)
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Identity layer.
    case_simple()
    # Cast input tensor into FLOAT32 / FLOAT16 / INT32 / INT64 / UINT8 / INT4 / BOOL
    case_datatype_conversion()
    # Cast input tensor into int8
    case_datatype_conversion_int8()  # deprecated
    print("Finish")
