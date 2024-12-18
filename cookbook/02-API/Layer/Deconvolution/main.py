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

from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

n_b, n_c, n_h, n_w = [1, 1, 3, 3]
n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
data = {"tensor": data}
w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    layer.num_output_maps = n_cout  # [Optional] Reset number of output channel later
    layer.kernel_size_nd = [n_hk, n_wk]  # [Optional] Reset size of convolution kernel later
    layer.kernel = trt.Weights(w)  # [Optional] Reset weight later
    layer.bias = trt.Weights(b)  # [Optional] Reset bias later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_stride_dilation_pad():
    nHStride, nWStride = 2, 2
    nHDilation, nWDilation = 2, 2
    nHPadding, nWPadding = 1, 1

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
    layer.stride_nd = [nHStride, nWStride]
    layer.dilation_nd = [nHDilation, nWDilation]
    layer.padding_nd = [nHPadding, nWPadding]
    layer.pre_padding = [nHPadding, nWPadding]
    layer.post_padding = [nHPadding, nWPadding]
    layer.padding_mode = trt.PaddingMode.SAME_UPPER

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_group():
    n_c1 = 2  # n_c in this example is 2
    n_group = 2
    n_cout = n_group
    data1 = {"tensor": np.tile(data["tensor"], [1, n_c1 // n_c, 1, 1])}
    w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))  # double the kernel as shape of [n_group, n_hk, n_wk]
    b1 = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b1))
    layer.num_groups = n_group

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

@case_mark
def case_3d():
    n_c1 = 2
    data1 = {"tensor": np.tile(data["tensor"], [1, n_c1 // n_c, 1, 1]).reshape([n_b, 1, n_c1, n_h, n_w])}
    w1 = np.ascontiguousarray(np.concatenate([w, -w], 0))

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w1), trt.Weights(b))

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

def case_int8qdq():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
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
    layer.set_input(1, layer_dq1.get_output(0))

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

    print("Finish")
