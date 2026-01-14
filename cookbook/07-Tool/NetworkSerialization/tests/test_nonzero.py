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

class TestNonZeroLayer:

    def test_case_simple(self, trt_cookbook_tester):

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

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
