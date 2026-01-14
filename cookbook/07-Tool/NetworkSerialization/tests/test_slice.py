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
import pytest
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt

class TestSliceLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data = {
                "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1])
            layer.mode = trt.SampleMode.WRAP  # [Optional] Modify slice mode

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_pad(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data = {
                "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant([1], np.array([-1], dtype=np.float32))  # Value of out-of-bound
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 2, 2])
            layer.mode = trt.SampleMode.FILL
            layer.set_input(4, layer1.get_output(0))

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_set_input(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data = {
                "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.int32))
            layer2 = tw.network.add_constant([4], np.array([1, 2, 3, 4], dtype=np.int32))
            layer3 = tw.network.add_constant([4], np.array([1, 1, 1, 1], dtype=np.int32))
            layer = tw.network.add_slice(tensor, [], [], [])
            layer.set_input(1, layer1.get_output(0))
            layer.set_input(2, layer2.get_output(0))
            layer.set_input(3, layer3.get_output(0))

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_shape_input(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data = {
                "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
                "tensor1": np.array([0, 0, 0, 0], dtype=np.int32),
                "tensor2": np.array([1, 2, 3, 4], dtype=np.int32),
                "tensor3": np.array([1, 1, 1, 1], dtype=np.int32),
            }

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            tensor3 = tw.network.add_input("tensor3", datatype_np_to_trt(data["tensor3"].dtype), data["tensor3"].shape)
            tw.profile.set_shape_input(tensor1.name, [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2])
            tw.profile.set_shape_input(tensor2.name, [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 4, 5])
            tw.profile.set_shape_input(tensor3.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
            tw.config.add_optimization_profile(tw.profile)
            layer = tw.network.add_slice(tensor0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
            layer.set_input(1, tensor1)
            layer.set_input(2, tensor2)
            layer.set_input(3, tensor3)

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    @pytest.mark.skip(reason="Skip test_case_dds in TestSliceLayer")
    def test_case_dds(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data = {
                "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
                np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
                np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
                np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
                "tensor1": np.array([1, 2, 3, 4], dtype=np.int32),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), [-1 for _ in data["tensor1"].shape])  # tensor1 is a execution input tensor
            tw.profile.set_shape(tensor1.name, data["tensor1"].shape, data["tensor1"].shape, data["tensor1"].shape)
            tw.config.add_optimization_profile(tw.profile)

            layer1 = tw.network.add_elementwise(tensor1, tensor1, trt.ElementWiseOperation.SUM)  # Compute shape tensor from earlier layer
            layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1])
            layer.set_input(2, layer1.get_output(0))

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)
