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
from tensorrt_cookbook import TRTWrapperV1, case_mark, pack_int4, check_api_coverage

@case_mark
def case_simple():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    layer = tw.network.add_constant(data["tensor"].shape, trt.Weights(np.ascontiguousarray(data["tensor"])))
    # - Input: no
    # - Outputs: T[shape0]
    # - Data Type: T in [bool, int4, int8, int32, int64, float8, float16, float32, bfloat16]
    # - Shape: shape0 is determined at build time.
    # np.ascontiguousarray() must be used while converting np.array to trt.Weights
    layer.shape = data["tensor"].shape  # Reset later
    layer.weights = trt.Weights(np.ascontiguousarray(data["tensor"]))  # Reset later

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

@case_mark
def case_datatype_int4():
    data = {
        "tensor": np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [-1, -2, -3, -4, -5, -6, -7, -8],
            [7, 6, 5, 4, 3, 2, 1, 0],
            [-7, -6, -5, -4, -3, -2, -1, 0],
        ], dtype=np.int8)
    }

    tw = TRTWrapperV1()
    data_packed = pack_int4(data["tensor"])
    layer = tw.network.add_constant(data["tensor"].shape, weights=trt.Weights(trt.int4, data_packed.ctypes.data, data["tensor"].size))
    # Quantized weights must be followed by a DQ node
    layer1 = tw.network.add_constant(shape=(), weights=np.ones(shape=(1), dtype=np.float32))
    layer = tw.network.add_dequantize(layer.get_output(0), layer1.get_output(0), trt.float32)

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

if __name__ == "__main__":
    # A simple case of using a constant layer.
    case_simple()
    # Use data type int4
    case_datatype_int4()

    print("Finish")
