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

import pytest
from pathlib import Path

import numpy as np
from tensorrt_cookbook import TRTWrapperV2, datatype_np_to_trt, get_plugin, enable_plugin_hook, disable_plugin_hook

class TestPluginV3Layer:
    """
    b_enable_plugin_hook (S1): Whether to use cookbook plugin hook
    b_provide_plugin_info_dict (S2): Whether to provide a detailed plugin information dictionary for the serialization process
    b_provide_plugin_so (S3): Whether to provide the plugin binary (.so file)

    |  No.  |  S1   |  S2   |  S3   |                Description                |             Solution              |
    | :---: | :---: | :---: | :---: | :---------------------------------------: | :-------------------------------: |
    |   0   | False | True  | True  |         All information provided          |          Rebuild network          |
    |   1   | False | True  | False |                No .so file                |        Create dummy plugin        |
    |   2   | False | False | True  |      No plugin constructor arguments      |        Create dummy plugin        |
    |   3   | False | False | False |         No any plugin information         |        Create dummy plugin        |
    |   4   | True  | True  | True  |         All information provided          | Rebuild network (hook is useless) |
    |   5   | True  | True  | False |                No .so file                |        Create dummy plugin        |
    |   6   | True  | False | True  | Get plugin constructor arguments by hooks | Rebuild network (hook is useful)  |
    |   7   | True  | False | False |         No any plugin information         |        Create dummy plugin        |
    """

    @pytest.mark.parametrize("b_provide_plugin_so", [True, False])
    @pytest.mark.parametrize("b_provide_plugin_info_dict", [True, False])
    @pytest.mark.parametrize("b_enable_plugin_hook", [True, False])
    def test_case_simple(self, b_enable_plugin_hook, b_provide_plugin_info_dict, b_provide_plugin_so, trt_cookbook_tester):

        def build_network(tw: TRTWrapperV2):

            if b_enable_plugin_hook:
                enable_plugin_hook()

            data = {"tensor": np.arange(60, dtype=np.float32).reshape([3, 4, 5])}
            plugin_info_dict = {
                "AddScalarPlugin_01": dict(
                    name="AddScalar",
                    version="1",
                    namespace="",
                    argument_dict={"scalar": np.array([1.0], dtype=np.float32)},
                    number_input_tensor=1,
                    number_input_shape_tensor=0,
                    plugin_api_version="3",
                    layer_name="AddScalarPlugin_01",
                )
            }

            tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1, -1, -1])
            tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
            tw.config.add_optimization_profile(tw.profile)

            layer = tw.network.add_plugin_v3([tensor], [], get_plugin(plugin_info_dict["AddScalarPlugin_01"]))
            layer.name = "AddScalarPlugin_01"
            tensor = layer.get_output(0)
            tensor.name = "tensor1"

            if b_enable_plugin_hook:
                disable_plugin_hook()

            return [layer.get_output(0)], data, {"plugin_info_dict": (plugin_info_dict if b_provide_plugin_info_dict else {})}

        b_create_dummy_plugin = not b_provide_plugin_so or (not b_provide_plugin_info_dict and not b_enable_plugin_hook)
        plugin_file_list = [Path("./pluginv3/AddScalarPlugin.so")] if b_provide_plugin_so else []

        assert trt_cookbook_tester(build_network, expect_fail_comparsion=b_create_dummy_plugin, plugin_file_list=plugin_file_list)
