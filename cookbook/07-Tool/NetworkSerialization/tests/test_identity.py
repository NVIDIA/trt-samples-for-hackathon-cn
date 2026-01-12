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

class TestIdentityLayer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            layer = tw.network.add_identity(tensor)

            return [layer.get_output(0)], data

        trt_cookbook_tester(build_network)

    def test_case_datatype_conversion(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

            tw.config.set_flag(trt.BuilderFlag.FP16)  # Needed if using float16
            tw.config.set_flag(trt.BuilderFlag.BF16)  # Needed if using bfloat16
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            output_tensor_list = []
            for data_type in [trt.float16, trt.int32, trt.int64, trt.uint8, trt.bool]:
                layer = tw.network.add_cast(tensor, data_type)
                layer.get_output(0).dtype = data_type
                output_tensor_list.append(layer.get_output(0))

            return output_tensor_list, data

        trt_cookbook_tester(build_network)

    def test_case_datatype_conversion_int8(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

            tw.config.set_flag(trt.BuilderFlag.INT8)  # Needed if using int8
            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
            output_tensor_list = []
            for data_type in [trt.int8]:
                layer = tw.network.add_cast(tensor, data_type)
                layer.get_output(0).set_dynamic_range(0, 127)  # dynamic range or calibration needed for INT8
                output_tensor_list.append(layer.get_output(0))

            return output_tensor_list, data

        trt_cookbook_tester(build_network)
