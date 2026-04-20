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
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, APIExcludeSet, grep_used_members

@case_mark
def case_plugin_v2():
    plugin_file = Path(__file__).parent / "AddScalarPluginV2.so"
    plugin_name = "AddScalar"

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(plugin_file)

    plugin_registry = trt.get_plugin_registry()
    instance_public_member = APIExcludeSet.analyze_public_members(plugin_registry, b_print=True)
    grep_used_members(Path(__file__), instance_public_member)

    plugin_creator = plugin_registry.get_plugin_creator(plugin_name, "1", "")  # Deprecated equivalent API, only works for plugin v2

    print(f"{plugin_creator.name = }")
    print(f"{plugin_creator.plugin_namespace = }")
    print(f"{plugin_creator.plugin_version = }")
    print(f"{plugin_creator.field_names = }")  # Its member can be got but not listed in dir()

    # Print the necessary parameters for creating the plugin
    for i, plugin_field in enumerate(plugin_creator.field_names):
        print(f"{i:2d}->{plugin_field.name},{plugin_field.type},{plugin_field.size},{plugin_field.data}")

    # Feed the PluginCreator with parameters
    plugin_field_collection = trt.PluginFieldCollection()
    plugin_field = trt.PluginField("scalar", np.float32(1.0), trt.PluginFieldType.FLOAT32)
    plugin_field_collection.append(plugin_field)  # Use like a list
    plugin = plugin_creator.create_plugin(plugin_creator.name, plugin_field_collection)

    plugin.__class__ = trt.IPluginV2DynamicExt  # member methods of trt.PluginV2, trt.PluginV2Ext, trt.IPluginV2DynamicExtBase are not showed in this example
    print(f"{plugin.FORMAT_COMBINATION_LIMIT =}")
    print(f"{plugin.plugin_type =}")
    print(f"{plugin.plugin_namespace =}")
    print(f"{plugin.plugin_version =}")
    print(f"{plugin.num_outputs =}")
    print(f"{plugin.serialization_size =}")
    print(f"{plugin.tensorrt_version =}")
    print(f"{plugin.get_serialization_size() =}")
    # APIs for plugin life cycle, not used in this example
    # plugin.clone()
    # plugin.initialize()
    # plugin.terminate()
    # plugin.destroy()
    # APIs not used in this example:
    # plugin.configure_plugin([trt.DynamicPluginTensorDesc()], [trt.DynamicPluginTensorDesc()])
    # plugin.configure_with_format([trt.Dims()], [trt.Dims()], trt.float32, trt.TensorFormat.LINEAR, 1)
    # plugin.enqueue([trt.PluginTensorDesc()], [trt.PluginTensorDesc()], [0], [0], 0, 0)
    # plugin.execute_async(1, [], [], None, 0)
    # plugin.get_output_datatype(0, [trt.float32])
    # plugin.get_output_dimensions(0, [trt.DimsExprs()], trt.IExprBuilder())
    # plugin.get_output_shape(0, [trt.Dims()])
    # plugin.get_workspace_size([trt.PluginTensorDesc()], [trt.PluginTensorDesc()])
    # plugin.supports_format(trt.float32, trt.TensorFormat.LINEAR)
    # plugin.supports_format_combination(0, [trt.PluginTensorDesc()], 1)

    # Serialization / Deserilization of the plugin parameters
    plugin_string = plugin.serialize()
    plugin = plugin_creator.deserialize_plugin(plugin_creator.name, plugin_string)  # create a plugin by memory of serialized plugin

    tw = TRTWrapperV1(logger=logger)
    tensor = tw.network.add_input("tensor", trt.float32, [-1])
    tw.profile.set_shape(tensor.name, [1], [2], [4])
    tw.config.add_optimization_profile(tw.profile)
    pluginLayer = tw.network.add_plugin_v2([tensor], plugin)
    print(pluginLayer.plugin)  # It is a clone of the input `plugin` argument above

if __name__ == "__main__":

    # List APIs for plugin v2, only shows the different (might be deprecated) APIs
    case_plugin_v2()

    print("Finish")
