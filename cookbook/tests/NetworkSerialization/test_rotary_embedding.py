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
from tensorrt_cookbook import TRTWrapperV2, datatype_cast

#@pytest.mark.skip(reason="TODO")
class TestRotaryEmbeddingLayer:

    def datatype_cast(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            b, n, s, h = 1, 2, 4, 8
            max_position = 16
            data = {
                "input": np.arange(b * n * s * h, dtype=np.float32).reshape(b, n, s, h) / 32,
                "cos_cache": np.cos(np.linspace(0, 1, max_position * (h // 2), dtype=np.float32)).reshape(max_position, h // 2),
                "sin_cache": np.sin(np.linspace(0, 1, max_position * (h // 2), dtype=np.float32)).reshape(max_position, h // 2),
                "position_ids": np.tile(np.arange(s, dtype=np.int64), (b, 1)),
            }

            input_tensor = tw.network.add_input("input", datatype_cast(data["input"].dtype, "trt"), data["input"].shape)
            cos_cache = tw.network.add_input("cos_cache", datatype_cast(data["cos_cache"].dtype, "trt"), data["cos_cache"].shape)
            sin_cache = tw.network.add_input("sin_cache", datatype_cast(data["sin_cache"].dtype, "trt"), data["sin_cache"].shape)
            position_ids = tw.network.add_input("position_ids", datatype_cast(data["position_ids"].dtype, "trt"), data["position_ids"].shape)

            layer = tw.network.add_rotary_embedding(input_tensor, cos_cache, sin_cache, False, 0)
            layer.set_input(3, position_ids)
            layer.metadata = "regression-rotary"
            layer.num_ranks = 1
            output_tensor = layer.get_output(0)
            output_tensor.name = "output"

            return [output_tensor], data

        assert trt_cookbook_tester(build_network)
