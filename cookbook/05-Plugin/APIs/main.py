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
from tensorrt_cookbook import TRTWrapperV1
import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark

@case_mark
def case_plugin_v3():
    plugin_file_list = [Path(__file__).parent / "AddScalarPluginV3.so"]
    plugin_name = "AddScalar"

    # Load default plugin creators
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    # trt.IPluginRegistry related
    plugin_registry = trt.get_plugin_registry()
    print(f"Count of default plugin creators = {len(plugin_registry.plugin_creator_list)}")

    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.DLA_STANDALONE) = }")
    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.SAFETY) = }")
    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD) = }")  # Only standard builder has plugin creators, the same as `plugin_registry` above

    # Attributions of Plugin Registry
    print(f"{plugin_registry.error_recorder = }")  # Get / set error recorder to the plugin register, default value: None
    print(f"{plugin_registry.parent_search_enabled = }")  # Get / Set whether search plugin creators in parent directory, default value: True

    # Print information of all plugin creators
    print("-" * 71)
    print(f"{'Plugin Name':^52s}|{'NameSpace':9s}|{'Version':7s}|")
    print("-" * 71)
    creator_list = [[plugin_creator.name, plugin_creator.plugin_namespace, plugin_creator.plugin_version] for plugin_creator in plugin_registry.all_creators]
    # plugin_registry.all_creators_recursive  # Stronger API for plugin_registry.all_creators, enabling recursive search when parent_search_enabled == False
    # plugin_registry.plugin_creator_list  # Deprecated API for plugin_registry.all_creators, several new plugins can not be found in the list

    for name, namespace, version in sorted(creator_list, key=lambda x: x[0]):
        print(f"{name:52s}|{namespace:^9s}|{version:^7s}|")
    print("-" * 71)

    # Load local plugin creators
    for plugin_file in plugin_file_list:
        ctypes.cdll.LoadLibrary(plugin_file)
        # Equivalent API to register / deregister the plugin file, do not use this since plugin_registry becomes invalid after calling this
        # handle = plugin_registry.load_library(str(plugin_file))
        # plugin_registry.deregister_library(handle)

    plugin_creator = plugin_registry.get_creator(plugin_name, "1", "")
    # plugin_creator = plugin_registry.get_plugin_creator(plugin_name, "1", "")  # Deprecated API, only works for plugin v2

    # Register / Deregister a plugin creator respectively
    # plugin_registry.deregister_creator(plugin_creator)
    # plugin_registry.register_creator(plugin_creator)

    # APIs not used in this example:
    # plugin_registry.acquire_plugin_resource

    # trt.IPluginCreatorV3One related:
    print(f"{plugin_creator.api_language = }")
    print(f"{plugin_creator.interface_info = }")
    print(f"{plugin_creator.interface_info.kind = }")
    print(f"{plugin_creator.interface_info.major = }")
    print(f"{plugin_creator.interface_info.minor = }")
    print(f"{plugin_creator.name = }")
    print(f"{plugin_creator.plugin_namespace = }")
    print(f"{plugin_creator.plugin_version = }")

    # Print the necessary parameters for creating the plugin
    for i, pluginField in enumerate(plugin_creator.field_names):
        print(f"{i:2d}->{pluginField.name},{pluginField.type},{pluginField.size},{pluginField.data}")

    # Feed the PluginCreator with parameters
    pluginFieldCollection = trt.PluginFieldCollection()
    pluginField = trt.PluginField("scalar", np.float32(1.0), trt.PluginFieldType.FLOAT32)
    # tensorrt.PluginFieldType: BF16, CHAR, DIMS, FLOAT16, FLOAT32, FLOAT64, FP4, FP8, INT16, INT32, INT4, INT64, INT8, UNKNOWN

    pluginFieldCollection.append(pluginField)  # Use like a list
    #pluginFieldCollection.insert(1,pluginField)
    #pluginFieldCollection.extend([pluginField])
    #pluginFieldCollection.clear()
    #pluginFieldCollection.pop(1)

    # Create a plugin from plugin creator
    plugin = plugin_creator.create_plugin(plugin_creator.name, pluginFieldCollection, trt.TensorRTPhase.BUILD)
    # plugin = plugin_creator.create_plugin(plugin_creator.name, pluginFieldCollection)  # Deprecated equivalent API for plugin v2

    print(f"{plugin.api_language =}")
    # APIs for plugin life cycle
    # plugin.clone()
    # plugin.destroy()

    interface_core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    print(f"{interface_core.api_language = }")
    print(f"{interface_core.interface_info = }")
    print(f"{interface_core.interface_info.kind = }")
    print(f"{interface_core.interface_info.major = }")
    print(f"{interface_core.interface_info.minor = }")
    print(f"{interface_core.plugin_name = }")
    print(f"{interface_core.plugin_namespace = }")
    print(f"{interface_core.plugin_version = }")

    interface_build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    print(f"{interface_build.DEFAULT_FORMAT_COMBINATION_LIMIT = }")
    print(f"{interface_build.format_combination_limit = }")
    print(f"{interface_build.interface_info = }")
    print(f"{interface_build.interface_info.kind = }")
    print(f"{interface_build.interface_info.major = }")
    print(f"{interface_build.interface_info.minor = }")
    print(f"{interface_build.metadata_string = }")
    print(f"{interface_build.num_outputs = }")
    print(f"{interface_build.timing_cache_id = }")
    # APIs not work here:
    interface_build.configure_plugin
    interface_build.format_combination_limit
    interface_build.get_aliased_input
    interface_build.get_output_data_types
    interface_build.get_output_shapes
    interface_build.get_valid_tactics
    interface_build.get_workspace_size
    interface_build.supports_format_combination

    interface_runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    print(f"{interface_runtime.api_language = }")
    print(f"{interface_runtime.interface_info = }")
    print(f"{interface_runtime.interface_info.kind = }")
    print(f"{interface_runtime.interface_info.major = }")
    print(f"{interface_runtime.interface_info.minor = }")
    # APIs not work here:
    interface_runtime.get_fields_to_serialize
    interface_runtime.attach_to_context
    interface_runtime.enqueue
    interface_runtime.get_fields_to_serialize
    interface_runtime.on_shape_change
    interface_runtime.set_tactic

    # Other classes
    plugin_tensor_desc = trt.PluginTensorDesc()
    print(f"{plugin_tensor_desc.dims = }")
    print(f"{plugin_tensor_desc.format = }")
    print(f"{plugin_tensor_desc.scale = }")
    print(f"{plugin_tensor_desc.type = }")

    dynamic_plugin_tensor_desc = trt.DynamicPluginTensorDesc()
    print(f"{dynamic_plugin_tensor_desc.desc = }")
    print(f"{dynamic_plugin_tensor_desc.min = }")
    print(f"{dynamic_plugin_tensor_desc.max = }")

    trt.IDimensionExpr  # This class has no constructor defined
    trt.IPluginResourceContext  # This class has no constructor defined!
    trt.DimsExprs()
    trt.IExprBuilder()
    trt.DimensionOperation
    # trt.DimensionOperation: CEIL_DIV, EQUAL, FLOOR_DIV, LESS, MAX, MIN, PROD, SUB, SUM

    tw = TRTWrapperV1(logger=logger)
    tensor = tw.network.add_input("tensor", trt.float32, [-1])
    tw.profile.set_shape(tensor.name, [1], [2], [4])
    pluginLayer = tw.network.add_plugin_v3([tensor], [], plugin)
    print(pluginLayer.plugin)  # It is a clone of the input `plugin` argument above

@case_mark
def case_plugin_v2():
    plugin_file_list = [Path(__file__).parent / "AddScalarPluginV2.so"]
    plugin_name = "AddScalar"

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    for plugin_file in plugin_file_list:
        ctypes.cdll.LoadLibrary(plugin_file)

    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(plugin_name, "1", "")  # Deprecated equivalent API, only works for plugin v2

    # trt.IPluginCreator related:
    print(f"{plugin_creator.api_language = }")
    print(f"{plugin_creator.interface_info = }")
    print(f"{plugin_creator.interface_info.kind = }")
    print(f"{plugin_creator.interface_info.major = }")
    print(f"{plugin_creator.interface_info.minor = }")
    print(f"{plugin_creator.name = }")
    print(f"{plugin_creator.plugin_namespace = }")
    print(f"{plugin_creator.plugin_version = }")
    print(f"{plugin_creator.field_names = }")  # Its member methods can be called but not listed in dir()

    # Print the necessary parameters for creating the plugin
    for i, pluginField in enumerate(plugin_creator.field_names):
        print(f"{i:2d}->{pluginField.name},{pluginField.type},{pluginField.size},{pluginField.data}")

    # Feed the PluginCreator with parameters
    pluginFieldCollection = trt.PluginFieldCollection()
    pluginField = trt.PluginField("scalar", np.float32(1.0), trt.PluginFieldType.FLOAT32)
    pluginFieldCollection.append(pluginField)  # Use like a list
    plugin = plugin_creator.create_plugin(plugin_creator.name, pluginFieldCollection)

    plugin.__class__ = trt.IPluginV2DynamicExt  # member methods of trt.PluginV2, trt.PluginV2Ext, trt.IPluginV2DynamicExtBase are not showed in this example
    print(f"{plugin.FORMAT_COMBINATION_LIMIT =}")
    print(f"{plugin.plugin_type =}")
    print(f"{plugin.plugin_namespace =}")
    print(f"{plugin.plugin_version =}")
    print(f"{plugin.num_outputs =}")
    print(f"{plugin.serialization_size =}")
    print(f"{plugin.tensorrt_version =}")
    print(f"{plugin.get_serialization_size() =}")
    # APIs for plugin life cycle
    # plugin.clone()
    # plugin.initialize()
    # plugin.terminate()
    # plugin.destroy()
    # Other useless APIs
    plugin.configure_plugin([trt.DynamicPluginTensorDesc()], [trt.DynamicPluginTensorDesc()])
    plugin.configure_with_format([trt.Dims()], [trt.Dims()], trt.float32, trt.TensorFormat.LINEAR, 1)
    plugin.enqueue([trt.PluginTensorDesc()], [trt.PluginTensorDesc()], [0], [0], 0, 0)
    plugin.execute_async(1, [], [], None, 0)
    plugin.get_output_datatype(0, [trt.float32])
    plugin.get_output_dimensions(0, [trt.DimsExprs()], trt.IExprBuilder())
    plugin.get_output_shape(0, [trt.Dims()])
    plugin.get_workspace_size([trt.PluginTensorDesc()], [trt.PluginTensorDesc()])
    plugin.supports_format(trt.float32, trt.TensorFormat.LINEAR)
    plugin.supports_format_combination(0, [trt.PluginTensorDesc()], 1)

    # Serialization / Deserilization of the plugin
    plugin_string = plugin.serialize()
    plugin = plugin_creator.deserialize_plugin(plugin_creator.name, plugin_string)  # create a plugin by memory of serialized plugin

    tw = TRTWrapperV1(logger=logger)
    tensor = tw.network.add_input("tensor", trt.float32, [-1])
    tw.profile.set_shape(tensor.name, [1], [2], [4])
    pluginLayer = tw.network.add_plugin_v2([tensor], plugin)
    print(pluginLayer.plugin)  # It is a clone of the input `plugin` argument above

if __name__ == "__main__":

    # List APIs for plugin v3
    case_plugin_v3()
    # List APIs for plugin v2, only shows the different (might be deprecated) APIs
    case_plugin_v2()

    print("Finish")
