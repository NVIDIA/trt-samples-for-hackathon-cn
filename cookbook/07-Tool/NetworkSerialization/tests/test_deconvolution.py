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
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt

class TestDeconvolutionLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            n_b, n_c, n_h, n_w = [1, 1, 3, 3]
            n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
            data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
            data = {"tensor": data}
            w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
            b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_stride_dilation_pad(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            n_b, n_c, n_h, n_w = [1, 1, 3, 3]
            n_cout, n_hk, n_wk = [1, 3, 3]
            nHStride, nWStride = 2, 2
            nHDilation, nWDilation = 2, 2
            nHPadding, nWPadding = 1, 1
            data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
            data = {"tensor": data}
            w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
            b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            layer.stride_nd = [nHStride, nWStride]
            layer.dilation_nd = [nHDilation, nWDilation]
            layer.padding_nd = [nHPadding, nWPadding]
            layer.pre_padding = [nHPadding, nWPadding]
            layer.post_padding = [nHPadding, nWPadding]
            layer.padding_mode = trt.PaddingMode.SAME_UPPER

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_group(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
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

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout1, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))
            layer.num_groups = n_group

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_3d(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            n_b, n_c, n_h, n_w = [1, 1, 3, 3]
            n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
            n_c1 = 2
            data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
            data = np.tile(data, [1, n_c1 // n_c, 1, 1]).reshape([n_b, 1, n_c1, n_h, n_w])
            data = {"tensor": data}
            w = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk)
            w = np.ascontiguousarray(np.concatenate([w, -w], 0))
            b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_deconvolution_nd(tensor, n_cout, [n_hk, n_wk], trt.Weights(w), trt.Weights(b))

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_int8qdq(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            n_b, n_c, n_h, n_w = [1, 1, 3, 3]
            n_cout, n_hk, n_wk = [1, 3, 3]  # Number of output channel, kernel height and kernel width
            data = np.arange(np.prod(n_b * n_c * n_h * n_w), dtype=np.float32).reshape(n_b, n_c, n_h, n_w) + 1
            data = {"tensor": data}
            w = np.ascontiguousarray(np.power(10, range(4, -5, -1), dtype=np.float32).reshape(n_cout, n_c, n_hk, n_wk))
            b = np.ascontiguousarray(np.zeros(n_cout, dtype=np.float32))

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

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)
