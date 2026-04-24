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

import ctypes
import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperV1, build_mnist_network_trt, case_mark, get_plugin)

@case_mark
def case_mnist():
    tw = TRTWrapperV1()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)
    tw.serialize_engine("model-mnist.trt")

@case_mark
def case_pluginv2():
    shape = [3, 4, 5]
    plugin_file_list = [Path(__file__).parent / "AddScalarPluginV2.so"]

    tw = TRTWrapperV1()

    # Use the deprecated code in `class TRTWrapperV1` for static plugin loading
    trt.init_libnvinfer_plugins(tw.logger, namespace="")
    for plugin_file in plugin_file_list:
        if plugin_file.exists():
            ctypes.cdll.LoadLibrary(plugin_file)

    plugin_info_dict = {
        "AddScalarPluginLayer": dict(
            name="AddScalar",
            version="1",
            namespace="",
            argument_dict=dict(scalar=np.array([1.0], dtype=np.float32)),
            number_input_tensor=1,
            number_input_shape_tensor=0,
            plugin_api_version="2",
        )
    }

    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_plugin_v2([input_tensor], get_plugin(plugin_info_dict["AddScalarPluginLayer"]))
    layer.name = "AddScalarPluginLayer"
    tensor = layer.get_output(0)
    tensor.name = "outputT0"

    tw.build([tensor])
    tw.serialize_engine("model-pluginv2.trt")

@case_mark
def case_pluginv3():
    shape = [3, 4, 5]
    plugin_file_list = [Path(__file__).parent / "AddScalarPluginV3.so"]

    tw = TRTWrapperV1(plugin_file_list=plugin_file_list)

    plugin_info_dict = {
        "AddScalarPluginLayer": dict(
            name="AddScalar",
            version="1",
            namespace="",
            argument_dict=dict(scalar=np.array([1.0], dtype=np.float32)),
            number_input_tensor=1,
            number_input_shape_tensor=0,
            plugin_api_version="3",
        )
    }

    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_plugin_v3([input_tensor], [], get_plugin(plugin_info_dict["AddScalarPluginLayer"]))
    layer.name = "AddScalarPluginLayer"
    tensor = layer.get_output(0)
    tensor.name = "outputT0"

    tw.build([tensor])
    tw.serialize_engine("model-pluginv3.trt")

if __name__ == "__main__":
    if os.getenv("CASE") == "mnist":
        case_mnist()
    elif os.getenv("CASE") == "pluginv2":
        case_pluginv2()
    elif os.getenv("CASE") == "pluginv3":
        case_pluginv3()
    else:
        raise ValueError("Set environment variable `CASE` in ['mnist', 'pluginv2', 'pluginv3']")

    print("Finish")
