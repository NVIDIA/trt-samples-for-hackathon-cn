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
from packaging.version import Version
from tensorrt_cookbook import TRTWrapperV2, datatype_cast

TRT_VERSION_GE_10_15 = Version(trt.__version__) >= Version("10.15")

class TestDynamicQuantizeLayer:

    def test_case_v1(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

            tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
            layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP8, trt.DataType.FLOAT)
            cast_layer = tw.network.add_identity(layer.get_output(0))  # Add one more layer to avoid FP8 network output
            cast_layer.get_output(0).dtype = trt.DataType.FLOAT

            return [cast_layer.get_output(0), layer.get_output(1)], data

        assert trt_cookbook_tester(build_network)

    def test_case_v1_double_quantization(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

            tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
            double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
            layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
            layer.set_input(1, double_quantization_layer.get_output(0))
            cast_layer = tw.network.add_identity(layer.get_output(0))  # Add one more layer to avoid FP8 network output
            cast_layer.get_output(0).dtype = trt.DataType.FLOAT
            cast_layer2 = tw.network.add_identity(layer.get_output(1))  # Add one more layer to avoid FP8 network output
            cast_layer2.get_output(0).dtype = trt.DataType.FLOAT

            return [cast_layer.get_output(0), cast_layer2.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    @pytest.mark.skipif(not TRT_VERSION_GE_10_15, reason="requires TensorRT >= 10.15")
    def test_case_v2(self, trt_cookbook_tester):  # Since TensrRT-10.15

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": (np.arange(64, dtype=np.float32)).reshape(8, 8) / 32 - 1}

            tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
            layer = tw.network.add_dynamic_quantize_v2(tensor, trt.Dims([4, 4]), trt.DataType.FP8, trt.DataType.FLOAT)
            cast_layer = tw.network.add_identity(layer.get_output(0))  # Add one more layer to avoid FP8 network output
            cast_layer.get_output(0).dtype = trt.DataType.FLOAT

            return [cast_layer.get_output(0), layer.get_output(1)], data

        assert trt_cookbook_tester(build_network)

    @pytest.mark.skipif(not TRT_VERSION_GE_10_15, reason="requires TensorRT >= 10.15")
    def test_case_v2_double_quantization(self, trt_cookbook_tester):  # Since TensrRT-10.15

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": (np.arange(64, dtype=np.float32)).reshape(8, 8) / 32 - 1}

            tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
            double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
            layer = tw.network.add_dynamic_quantize_v2(tensor, trt.Dims([4, 4]), trt.DataType.FP4, trt.DataType.FP8)
            layer.set_input(1, double_quantization_layer.get_output(0))
            cast_layer = tw.network.add_identity(layer.get_output(0))  # Add one more layer to avoid FP8 network output
            cast_layer.get_output(0).dtype = trt.DataType.FLOAT
            cast_layer2 = tw.network.add_identity(layer.get_output(1))  # Add one more layer to avoid FP8 network output
            cast_layer2.get_output(0).dtype = trt.DataType.FLOAT

            return [cast_layer.get_output(0), cast_layer2.get_output(0)], data

        assert trt_cookbook_tester(build_network, expect_fail_building=True)
