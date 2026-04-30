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
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2, datatype_cast

class TestKVCacheUpdateLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            b, n, s_max, s_new, h = 1, 2, 8, 2, 8
            data = {
                "cache": np.zeros((b, n, s_max, h), dtype=np.float16),
                "update": np.arange(b * n * s_new * h, dtype=np.float16).reshape(b, n, s_new, h) / 16,
                "write_indices": np.array([0], dtype=np.int32),
            }

            tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

            cache = tw.network.add_input("cache", datatype_cast(data["cache"].dtype, "trt"), data["cache"].shape)
            update = tw.network.add_input("update", datatype_cast(data["update"].dtype, "trt"), data["update"].shape)
            write_indices = tw.network.add_input("write_indices", datatype_cast(data["write_indices"].dtype, "trt"), data["write_indices"].shape)

            layer = tw.network.add_kv_cache_update(cache, update, write_indices, trt.KVCacheMode.LINEAR)
            layer.metadata = "regression-kv"
            layer.num_ranks = 1
            output_tensor = layer.get_output(0)
            output_tensor.name = "cache_out"

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
