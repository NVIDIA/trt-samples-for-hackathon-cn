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

class TestEinsumLayer:

    def test_case_contraction(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4),
                "tensor1": np.arange(np.prod(30), dtype=np.float32).reshape(2, 3, 5),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], "ijk,pjr->ikpr")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_transpose(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_einsum([tensor], "ijk->jki")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_sum_reduce(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_einsum([tensor], "ijk->ij")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_dot_product(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            if True:
                shape0 = 1, 1, 4
                shape1 = 1, 1, 4
                equation = "ijk,pqk->"
            elif True:  # Alternative example 1
                shape0 = 1, 2, 4
                shape1 = 1, 3, 4
                equation = "ijk,pqk->"
            else:  # Alternative example 2
                shape0 = 1, 2, 4
                shape1 = 1, 3, 4
                equation = "ijk,pqk->j"
            data = {
                "tensor": np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0),
                "tensor1": np.ones(np.prod(shape1), dtype=np.float32).reshape(shape1),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], equation)

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_matrix_multiplication(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.arange(np.prod(12), dtype=np.float32).reshape(2, 2, 3),
                "tensor1": np.ones(np.prod(24), dtype=np.float32).reshape(2, 3, 4),
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
            layer = tw.network.add_einsum([tensor, tensor1], "ijk,ikl->ijl")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_multi_tensor_contraction(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {
                "tensor": np.arange(np.prod(6), dtype=np.float32).reshape(1, 2, 3),
                "tensor1": np.ones(np.prod(24), dtype=np.float32).reshape(4, 3, 2),
                "tensor2": np.ones(np.prod(20), dtype=np.float32).reshape(4, 5),
            }

            tensor = tw.network.add_input("tensor", trt.float32, [1, 2, 3])
            tensor1 = tw.network.add_input("tensor1", trt.float32, [4, 3, 2])
            tensor2 = tw.network.add_input("tensor2", trt.float32, [4, 5])
            layer = tw.network.add_einsum([tensor, tensor1, tensor2], "abc,dcb,de->ae")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_diagnal(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(16), dtype=np.float32).reshape(1, 4, 4)}

            tensor = tw.network.add_input("tensor", trt.float32, [1, 4, 4])
            layer = tw.network.add_einsum([tensor], "ijj->ij")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)

    def test_case_ellipsis(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(24), dtype=np.float32).reshape(2, 3, 4)}

            tensor = tw.network.add_input("tensor", trt.float32, [1, 3, 4])
            layer = tw.network.add_einsum([tensor], "...j->...j")

            return [layer.get_output(0)], data

        assert trt_cookbook_tester(build_network)
