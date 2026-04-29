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
class TestKVCacheUpdateLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.zeros([31193], dtype=np.float32)}

            cache = tw.network.add_input("cache", trt.float16, [1, 2, 8, 8])
            update = tw.network.add_input("update", trt.float16, [1, 2, 2, 8])
            write_indices = tw.network.add_input("write_indices", trt.int32, [1])

            layer = tw.network.add_kv_cache_update(cache, update, write_indices, trt.KVCacheMode.LINEAR)
            layer.metadata = "regression-kv"
            layer.num_ranks = 1
            output_tensor = layer.get_output(0)
            output_tensor.name = "cache_out"
            tw.network.mark_output(output_tensor)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
