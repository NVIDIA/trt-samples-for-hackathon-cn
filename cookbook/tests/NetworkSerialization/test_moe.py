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

import numpy as np
import pytest
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2

@pytest.mark.skip(reason="TODO")
class TestMoELayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.zeros([31193], dtype=np.float32)}

            hidden_states = tw.network.add_input("hidden_states", trt.float16, [1, 2, 8])
            selected_experts = tw.network.add_input("selected_experts_for_tokens", trt.int32, [1, 2, 1])
            scores = tw.network.add_input("scores_for_selected_experts", trt.float16, [1, 2, 1])

            layer = tw.network.add_moe(hidden_states, selected_experts, scores)
            layer.activation_type = trt.MoEActType.SILU
            layer.metadata = "regression-moe"
            layer.num_ranks = 1
            output_tensor = layer.get_output(0)
            output_tensor.name = "output"
            tw.network.mark_output(output_tensor)

            return [output_tensor], data

        assert trt_cookbook_tester(build_network)
