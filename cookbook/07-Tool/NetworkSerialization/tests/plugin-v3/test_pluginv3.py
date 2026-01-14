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

from pathlib import Path

import numpy as np
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt, get_plugin_v3

class TestPluginV3Layer:

    def test_case_simple(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape([3, 4, 5])}
            plugin_info_dict = {
                "AddScalarPlugin_01": {
                    "name": "AddScalar",
                    "version": "1",
                    "namespace": "",
                    "argument_dict": {
                        "scalar": np.array([1.0], dtype=np.float32)
                    },
                    "number_input_tensor": 1,
                    "number_input_shape_tensor": 0,
                },
            }
            # TODO: mark number of input tensor and input shape tensor in plugin v3

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1, -1, -1])
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
            tw.config.add_optimization_profile(tw.profile)

            layer = tw.network.add_plugin_v3([tensor], [], get_plugin_v3(plugin_info_dict["AddScalarPlugin_01"]))
            layer.name = "AddScalarPlugin_01"
            tensor = layer.get_output(0)
            tensor.name = "tensor1"

            return [layer.get_output(0)], data, {"plugin_info_dict": plugin_info_dict}

        assert trt_cookbook_tester(build_network, plugin_file_list=[Path("./plugin-v3/AddScalarPlugin.so")])

    def test_case_dummy(self, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.arange(60, dtype=np.float32).reshape([3, 4, 5])}
            plugin_info_dict = {
                "AddScalarPlugin_01": {
                    "name": "AddScalar",
                    "version": "1",
                    "namespace": "",
                    "argument_dict": {
                        "scalar": np.array([1.0], dtype=np.float32)
                    },
                    "number_input_tensor": 1,
                    "number_input_shape_tensor": 0,
                },
            }
            # TODO: mark number of input tensor and input shape tensor in plugin v3

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1, -1, -1])
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
            tw.config.add_optimization_profile(tw.profile)

            layer = tw.network.add_plugin_v3([tensor], [], get_plugin_v3(plugin_info_dict["AddScalarPlugin_01"]))
            layer.name = "AddScalarPlugin_01"
            tensor = layer.get_output(0)
            tensor.name = "tensor1"

            return [layer.get_output(0)], data, {"plugin_info_dict": {}}

        assert trt_cookbook_tester(build_network, expect_fail_comparsion=True, plugin_file_list=[Path("./plugin-v3/AddScalarPlugin.so")])
