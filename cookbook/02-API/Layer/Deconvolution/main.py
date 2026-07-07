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
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = {"tensor": data}
    w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    # Input: input: T[shape0], weight: T[shape1], bias: T[shape2],
    # Output: T[shape3]
    # Data Type: T in [float16, float32, bfloat16, int8, float8]
    # Shape: len(shape0) in [4, 8], len(shape1) in [4, 5], len(shape2) in [1], np.prod(shape0) <= 2**31, np.prod(shape3) <= 2**31
    # If shape0 and shape3 is determined at build-time: np.prod(shape0) <= 2**40, np.prod(shape3) <= 2**40
    layer.num_output_maps = n_cout  # [Optional] Number of output feature maps; must be build-time constant
    # The number of output channel must be build-time constant (rather than -1).
    layer.kernel_size_nd = [n_hk, n_wk]  # [Optional] Kernel dimensions per spatial axis; 2 or 3 elements
    layer.kernel = trt.Weights(w)  # [Optional] Kernel weights; must match input data type
    layer.bias = trt.Weights(b)  # [Optional] Bias weights; must match input data type

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_stride_dilation_pad():
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]
    nHStride, nWStride = 2, 2
    nHDilation, nWDilation = 2, 2
    nHPadding, nWPadding = 1, 1
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = {"tensor": data}
    w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    layer.stride_nd = [nHStride, nWStride]  # [Optional] Default: [1, 1, ...]
    layer.dilation_nd = [nHDilation, nWDilation]  # [Optional] Default: [1, 1, ...]

    # Priority of padding APIs: padding_mode > pre_padding = post_padding > padding_nd
    layer.padding_nd = [nHPadding, nWPadding]  # [Optional] Default: [0, 0, ...]
    layer.pre_padding = [nHPadding, nWPadding]  # [Optional] Default: [0, 0, ...]
    layer.post_padding = [nHPadding, nWPadding]  # [Optional] Default: [0, 0, ...]
    layer.padding_mode = trt.PaddingMode.SAME_UPPER  # [Optional] Default: EXPLICIT_ROUND_DOWN; options: EXPLICIT_ROUND_DOWN, EXPLICIT_ROUND_UP, SAME_UPPER, SAME_LOWER

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_group():
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    n_cout1 = 2  # n_c in this example is 2
    n_group = 2
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = np.tile(data, [1, n_cout1 // n_c, 1, 1])
    data = {"tensor": data}
    w = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk)
    w = np.ascontiguousarray(np.concatenate([w, -w], 0))  # double the kernel as shape of [n_group, n_hk, n_wk]
    b = np.ascontiguousarray(np.zeros(n_cout1, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout1, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    # Both the channel count of input tensor and kernel must be able to be divided by the number of groups.
    # In int8 group convolution, the channel count in each group (nC/nGroup and nCOut/nGroup) should be multiple of 4.

    layer.num_groups = n_group  # [Optional] Default: 1; for int8, input and output channels per group must be multiple of 4

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_3d():
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    n_c1 = 2
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = np.tile(data, [1, n_c1 // n_c, 1, 1]).reshape([n_b, 1, n_c1, n_h, n_w])
    data = {"tensor": data}
    w = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk)
    w = np.ascontiguousarray(np.concatenate([w, -w], 0))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    # Rank of input tensor must be 5 or more.
    # Convolution kernel can move through dimension C.

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_int8qdq():
    n_b, n_c, n_h, n_w = [1, 1, 3, 3]
    n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
    data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
    data = {"tensor": data}
    w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
    b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer_q0_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q1_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_weight = tw.network.add_constant(w.shape, trt.Weights(w))
    layer_q0 = tw.network.add_quantize(tensor, layer_q0_weight.get_output(0))
    layer_q0.axis = 0
    layer_dq0 = tw.network.add_dequantize(layer_q0.get_output(0), layer_q1_weight.get_output(0))
    layer_dq0.axis = 0
    layer_q1 = tw.network.add_quantize(layer_weight.get_output(0), layer_q0_weight.get_output(0))
    layer_q1.axis = 0
    layer_dq1 = tw.network.add_dequantize(layer_q1.get_output(0), layer_q1_weight.get_output(0))
    layer_dq1.axis = 0
    layer = tw.network.add_deconvolution_nd(layer_dq0.get_output(0), n_cout, [n_hk, n_wk], trt.Weights(), trt.Weights(np.ascontiguousarray(b)))  # weight as empty
    layer.set_input(1, layer_dq1.get_output(0))  # Set weight from tensor rather than constructor

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using deconvolution layer
    case_simple()
    # Modify deconvolution parameters
    case_stride_dilation_pad()
    # A case of group deconvolution
    case_group()
    # A case of 3D deconvolution
    case_3d()
    # A case of QDQ-INT8 deconvolution with weights from another layer
    case_int8qdq()

    print_enumerated_members(trt.PaddingMode)

    print("Finish")
