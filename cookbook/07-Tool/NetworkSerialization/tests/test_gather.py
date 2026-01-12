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

class TestGatherLayer:

    def test_case_default_mode(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3]).reshape(1, 1, 1, shape[3])
            data = {
                "tensor": data0.astype(np.float32),
                "tensor1": np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)  # Index can be negetive
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
            layer.axis = 2

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_default_mode_num_elementwise_axis_1(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3]).reshape(1, 1, 1, shape[3])
            data = {
                "tensor": data0.astype(np.float32),
                "tensor1": np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
            layer.axis = 2
            layer.num_elementwise_dims = 1

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_element_mode(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3]).reshape(1, 1, 1, shape[3])

            data1 = np.zeros(data0.shape, dtype=np.int32)
            # use random permutation
            for i in range(data0.shape[0]):
                for j in range(data0.shape[1]):
                    for k in range(data0.shape[3]):
                        data1[i, j, :, k] = np.random.permutation(range(data0.shape[2]))
            data = {
                "tensor": data0.astype(np.float32),
                "tensor1": data1,
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ELEMENT)
            layer.axis = 2

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_nd_mode(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3]).reshape(1, 1, 1, shape[3])
            data = {
                "tensor": data0.astype(np.float32),
                "tensor1": np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_nd_mode_num_elementwise_axis_1(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [2, 3, 4, 5]
            data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3]).reshape(1, 1, 1, shape[3])
            data = {
                "tensor": data0.astype(np.float32),
                "tensor1": np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)
            layer.num_elementwise_dims = 1

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_gather_nonzeros(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = np.zeros([3, 4, 5]).astype(np.float32)
            data[0, 0, 1] = 1
            data[0, 2, 3] = 2
            data[0, 3, 4] = 3
            data[1, 1, 0] = 4
            data[1, 1, 1] = 5
            data[1, 1, 2] = 6
            data[1, 1, 3] = 7
            data[1, 1, 4] = 8
            data[2, 0, 1] = 9
            data[2, 1, 1] = 10
            data[2, 2, 1] = 11
            data[2, 3, 1] = 12
            data = {"tensor": data}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_non_zero(tensor)
            layer = tw.network.add_shuffle(layer.get_output(0))
            layer.first_transpose = [1, 0]
            layer = tw.network.add_gather_v2(tensor, layer.get_output(0), trt.GatherMode.ND)

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)
