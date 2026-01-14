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

class TestSelectLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 4, 5]
            data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            data = {
                "tensor": data0,
                "tensor1": -data0,
                "tensor2": (np.arange(np.prod(shape)) % 2).astype(bool).reshape(shape),
            }

            tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
            layer = tw.network.add_select(tensor2, tensor0, tensor1)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
