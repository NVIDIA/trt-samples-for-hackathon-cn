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
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt

class TestPaddingLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_padding_nd(tensor, [1, 2], [3, 4])

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_crop(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_padding_nd(tensor, [-1, 0], [0, -2])

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
