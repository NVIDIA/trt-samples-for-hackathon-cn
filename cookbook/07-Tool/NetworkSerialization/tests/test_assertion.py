#
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

import pytest
import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt

class TestAssertLayer:

    @pytest.mark.parametrize("b_can_pass", [True, False])
    def test_case_buildtime_check(self, b_can_pass, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.ones([3, 4, 5], dtype=np.float32)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer1 = tw.network.add_shape(tensor)
            layer2 = tw.network.add_slice(layer1.get_output(0), [2], [1], [1])
            if b_can_pass:
                layerConstant = tw.network.add_constant([1], np.array([5], dtype=np.int64))
            else:
                layerConstant = tw.network.add_constant([1], np.array([4], dtype=np.int64))
            layer3 = tw.network.add_elementwise(layer2.get_output(0), layerConstant.get_output(0), trt.ElementWiseOperation.EQUAL)
            layer4 = tw.network.add_identity(layer3.get_output(0))
            layer4.get_output(0).dtype = trt.bool
            layer = tw.network.add_assertion(layer4.get_output(0), "tensor.shape[2] != 5")
            return [layer4.get_output(0)], data

        trt_cookbook_tester(build_network, expect_fail_building=(not b_can_pass))

    @pytest.mark.parametrize("b_can_pass", [True, False])
    def test_case_runtime_check(self, b_can_pass, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.zeros([3, 4], dtype=np.float32)}
            data1 = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.zeros([3, 5], dtype=np.float32)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1, -1, -1])
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), [-1, -1])
            tw.profile.set_shape(tensor1.name, [1, 1], [3, 4], [6, 8])
            tw.config.add_optimization_profile(tw.profile)
            layer1 = tw.network.add_shape(tensor)
            layer2 = tw.network.add_slice(layer1.get_output(0), [1], [1], [1])
            layer3 = tw.network.add_shape(tensor1)
            layer4 = tw.network.add_slice(layer3.get_output(0), [1], [1], [1])
            # assert(tensor.shape[0] == tensor1.shape[0])
            layer5 = tw.network.add_elementwise(layer2.get_output(0), layer4.get_output(0), trt.ElementWiseOperation.EQUAL)
            # Assert layer seems no use but actually works
            tw.network.add_assertion(layer5.get_output(0), "[Something else we want to say]")
            layer6 = tw.network.add_cast(layer5.get_output(0), trt.int32)
            return [layer6.get_output(0)], data, {"runtime_data": (data if b_can_pass else data1)}

        trt_cookbook_tester(build_network, expect_exception=(None if b_can_pass else RuntimeError))
