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
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_transpose(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].shape  # No transpose compared with `case_simple`
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.TRANSPOSE)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_vector(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape[:-1]  # One less dimension compared with `case_simple`
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.VECTOR)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_broadcast(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            weight_shape = (1, 1) + data["tensor"].transpose(0, 1, 3, 2).shape[-2:]  # [1,1,5,4]
            layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
            layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
