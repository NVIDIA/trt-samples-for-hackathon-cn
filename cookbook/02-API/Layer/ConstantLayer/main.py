#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark

shape = [3, 4, 5]
data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}

data2 = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [-1, -2, -3, -4, -5, -6, -7, -8],
    [7, 6, 5, 4, 3, 2, 1, 0],
    [-7, -6, -5, -4, -3, -2, -1, 0],
], dtype=np.int8)

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

    layer = tw.network.add_constant(shape, trt.Weights(np.ascontiguousarray(data["inputT0"])))

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

@case_mark
def case_weight_shape():
    tw = TRTWrapperV1()

    layer = tw.network.add_constant([1], np.array([1], dtype=np.float32))
    layer.weights = trt.Weights(np.ascontiguousarray(data["inputT0"]))  # modify the data and its shape
    layer.shape = shape

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

@case_mark
def case_datatype_int4():
    tw = TRTWrapperV1()

    data2_packed = pack_int4(data2)
    layer = tw.network.add_constant(shape=data2.shape, weights=trt.Weights(trt.int4, data2_packed.ctypes.data, data2.size))
    # Quantized weights must be followed by a DQ node
    scale = tw.network.add_constant(shape=(), weights=np.ones(shape=(1), dtype=np.float32))
    layer_dq = tw.network.add_dequantize(layer.get_output(0), scale.get_output(0), trt.float32)
    layer_dq.precision = trt.int4

    tw.build([layer_dq.get_output(0)])
    tw.setup()
    tw.infer()

if __name__ == "__main__":
    case_simple()
    case_weight_shape()
    case_datatype_int4()

    print("Finish")
