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

class TestScatterLayer:

    def test_case_element_mode(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = 1, 3, 4, 5
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.tile(np.arange(shape[2], dtype=np.int32), [shape[0], shape[1], 1, shape[3]]).reshape(shape)
            data2 = -data0
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}
            scatter_axis = 2

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ELEMENT)
            layer.axis = scatter_axis
            layer.get_output(0).name = "outputT0"

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_nd_mode(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.array([[[0, 2, 1, 1], [1, 0, 3, 2], [0, 1, 2, 3]], [[1, 2, 1, 1], [0, 0, 3, 2], [1, 1, 2, 3]]], dtype=np.int32)
            data2 = -np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape[0], shape[1])
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)
            layer.get_output(0).name = "outputT0"

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_nd_mode_2(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data1 = np.array([[0, 2, 1], [1, 0, 3], [0, 1, 2], [1, 2, 1], [0, 0, 3], [1, 1, 2]], dtype=np.int32)
            data2 = -np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
            data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)
            layer.get_output(0).name = "outputT0"
            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
