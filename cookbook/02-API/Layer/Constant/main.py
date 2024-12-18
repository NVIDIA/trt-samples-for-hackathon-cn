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

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark

data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

data1 = {
    "tensor": np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [-1, -2, -3, -4, -5, -6, -7, -8],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [-7, -6, -5, -4, -3, -2, -1, 0],
    ], dtype=np.int8)
}

def pack_int4(array: np.ndarray):  # copy from https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/Constant.html
    result = []
    array = array.flatten()
    for low, high in zip(array[::2], array[1::2]):
        low = np.rint(np.clip(low, -8, 7)).astype(np.int8)
        high = np.rint(np.clip(high, -8, 7)).astype(np.int8)
        result.append(high << 4 | low & 0x0F)
    return np.asarray(result, dtype=np.int8)

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    layer = tw.network.add_constant(data["tensor"].shape, trt.Weights(np.ascontiguousarray(data["tensor"])))
    layer.weights = trt.Weights(np.ascontiguousarray(data["tensor"]))  # [Optional] Reset weight later
    layer.shape = data["tensor"].shape  # [Optional] Reset shape later

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

@case_mark
def case_datatype_int4():
    tw = TRTWrapperV1()

    data1_packed = pack_int4(data1["tensor"])
    layer = tw.network.add_constant(data1["tensor"].shape, weights=trt.Weights(trt.int4, data1_packed.ctypes.data, data1["tensor"].size))
    # Quantized weights must be followed by a DQ node
    layer1 = tw.network.add_constant(shape=(), weights=np.ones(shape=(1), dtype=np.float32))
    layer2 = tw.network.add_dequantize(layer.get_output(0), layer1.get_output(0), trt.float32)
    layer2.precision = trt.int4

    tw.build([layer2.get_output(0)])
    tw.setup()
    tw.infer()

if __name__ == "__main__":
    # A simple case of using a constant layer.
    case_simple()
    # Use data type INT4
    case_datatype_int4()

    print("Finish")
