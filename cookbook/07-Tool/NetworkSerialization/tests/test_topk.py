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

class TestTopKLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1)

            return [layer.get_output(0), layer.get_output(1)], data

        trt_cookbook_tester(build_network)

    def test_case_shape_input(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5),
                "tensor1": np.array([2], dtype=np.int32),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), [])
            tw.profile.set_shape_input(tensor1.name, [1], [2], [3])
            tw.config.add_optimization_profile(tw.profile)

            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
            layer.set_input(1, tensor1)

            return [layer.get_output(0), layer.get_output(1)], data

        trt_cookbook_tester(build_network)

    def test_case_shape_input(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5),
                "tensor1": np.array([3, -1], dtype=np.int32),  # tensor1 is a execution input tensor
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), [-1 for _ in data["tensor1"].shape])
            tw.profile.set_shape(tensor1.name, [1 for _ in data["tensor1"].shape], data["tensor1"].shape, data["tensor1"].shape)
            tw.config.add_optimization_profile(tw.profile)

            layer1 = tw.network.add_reduce(tensor1, trt.ReduceOperation.SUM, 1 << 0, False)  # Compute K from earlier layer
            layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
            layer.set_input(1, layer1.get_output(0))

            return [layer.get_output(0), layer.get_output(1)], data

        trt_cookbook_tester(build_network)
