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

class TestIfConditionStructure:

    @pytest.mark.parametrize("b_true_branch", [True, False])
    def test_case_simple(self, b_true_branch, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            if b_true_branch:
                data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}
            else:
                data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            # Extract the scalar first element
            layer1 = tw.network.add_shuffle(tensor)
            layer1.reshape_dims = [-1]
            layer2 = tw.network.add_slice(layer1.get_output(0), [0], [1], [1])
            layer3 = tw.network.add_shuffle(layer2.get_output(0))
            layer3.reshape_dims = []
            layer4 = tw.network.add_cast(layer3.get_output(0), trt.bool)

            if_structure = tw.network.add_if_conditional()
            if_structure.name = "A cute If Condition Structure"

            layer_input = if_structure.add_input(tensor)
            if_structure.set_condition(layer4.get_output(0))
            # Branch of condition is true
            layer_true = tw.network.add_elementwise(layer_input.get_output(0), layer_input.get_output(0), trt.ElementWiseOperation.SUM)
            # Branch of condition is false
            layer_false = tw.network.add_identity(layer_input.get_output(0))
            layer_output = if_structure.add_output(layer_true.get_output(0), layer_false.get_output(0))

            return [layer_output.get_output(0)], data

        trt_cookbook_tester(build_network)
