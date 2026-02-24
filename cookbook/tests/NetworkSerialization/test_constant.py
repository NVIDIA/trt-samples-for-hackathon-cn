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

class TestConstantLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

            layer = tw.network.add_constant(data["tensor"].shape, trt.Weights(np.ascontiguousarray(data["tensor"])))

            return [layer.get_output(0)], dict()

        assert trt_cookbook_tester(build_network)

    @pytest.mark.skip(reason="Skip test_case_datatype_int4 in TestConstantLayer")
    def test_case_datatype_int4(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.array([
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [-1, -2, -3, -4, -5, -6, -7, -8],
                    [7, 6, 5, 4, 3, 2, 1, 0],
                    [-7, -6, -5, -4, -3, -2, -1, 0],
                ], dtype=np.int8)
            }

            def pack_int4(array: np.ndarray):
                result = []
                array = array.flatten()
                for low, high in zip(array[::2], array[1::2]):
                    low = np.rint(np.clip(low, -8, 7)).astype(np.int8)
                    high = np.rint(np.clip(high, -8, 7)).astype(np.int8)
                    result.append(high << 4 | low & 0x0F)
                return np.asarray(result, dtype=np.int8)

            data_packed = pack_int4(data["tensor"])

            layer = tw.network.add_constant(data["tensor"].shape, weights=trt.Weights(trt.int4, data_packed.ctypes.data, data["tensor"].size))
            layer1 = tw.network.add_constant(shape=(), weights=np.ones(shape=(1), dtype=np.float32))
            layer2 = tw.network.add_dequantize(layer.get_output(0), layer1.get_output(0), trt.float32)

            return [layer2.get_output(0)], dict()

        assert trt_cookbook_tester(build_network)
