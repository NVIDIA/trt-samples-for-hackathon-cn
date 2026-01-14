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
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt

class TestActivationLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 3, 3]
            data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            scale = np.ascontiguousarray(np.array([0.5], dtype=np.float32))
            shift = np.ascontiguousarray(np.array([-7.0], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1.0], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.UNIFORM, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_channel(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 3, 3]
            data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
            scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_element(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 3, 3]
            data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.full(shape[1:], -7.0, dtype=np.float32))
            scale = np.ascontiguousarray(np.full(shape[1:], 0.5, dtype=np.float32))
            power = np.ascontiguousarray(np.ones(shape[1:], dtype=np.float32))
            layer = tw.network.add_scale(tensor, trt.ScaleMode.ELEMENTWISE, trt.Weights(shift), trt.Weights(scale), trt.Weights(power))

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_scale_channel_axis(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            shape = [1, 3, 3, 3]
            data = {"tensor": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            shift = np.ascontiguousarray(np.array([-2.5, -7.0, -11.5], dtype=np.float32))
            scale = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
            power = np.ascontiguousarray(np.array([1, 1, 1], dtype=np.float32))
            layer = tw.network.add_scale_nd(tensor, trt.ScaleMode.CHANNEL, trt.Weights(shift), trt.Weights(scale), trt.Weights(power), 0)
            layer.channel_axis = 1
            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
