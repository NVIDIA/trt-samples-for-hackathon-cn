# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperV1, case_mark, check_api_coverage, print_enumerated_members)

@case_mark
def case_plugin_v3():
    plugin_file = Path(__file__).parent / "AddScalarPluginV3.so"
    plugin_name = "AddScalar"

    # trt.IPluginRegistry related
    # `trt.get_plugin_registry()` returns standard runtime global plugin registry
    # `trt.get_builder_plugin_registry(capability)` returns certain registry with certain capability
    # `trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)` is equal to `trt.get_plugin_registry()`
    # SAFE and DLA_STANDALONE returns None

    plugin_registry = trt.get_plugin_registry()

    builder_plugin_registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)  # Refer to 05-Plugin/APIs
    print(f"trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD) = {builder_plugin_registry}")
    # Alternative values of argument:
    # trt.EngineCapability.STANDARD         -> 0
    # trt.EngineCapability.SAFETY           -> 1
    # trt.EngineCapability.DLA_STANDALONE   -> 2

    check_api_coverage(plugin_registry)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # plugin_registry.acquire_plugin_resource
    # plugin_registry.release_plugin_resource
    # plugin_registry._pybind11_conduit_v1()

    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.DLA_STANDALONE) = }")
    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.SAFETY) = }")
    print(f"{trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD) = }")  # Only standard builder has plugin creators, the same as `plugin_registry` above

    # Attributions of Plugin Registry
    print(f"{plugin_registry.error_recorder = }")  # Get / set error recorder to the plugin register, default: None
    print(f"{plugin_registry.parent_search_enabled = }")  # Get / Set whether search plugin creators in parent directory, default: True

    creator_list = [[plugin_creator.name, plugin_creator.plugin_namespace, plugin_creator.plugin_version] for plugin_creator in plugin_registry.all_creators_recursive]
    # Similar APIs:
    # plugin_registry.all_creators  # Disable recursive search when parent_search_enabled == False
    # plugin_registry.plugin_creator_list  # Deprecated, several new plugins can not be found in the list

    # Print information of all plugin creators
    print(f"Count of default plugin creators = {len(creator_list)}")
    print("-" * 71)
    print(f"{'Plugin Name':^52s}|{'NameSpace':9s}|{'Version':7s}|")
    print("-" * 71)
    for name, namespace, version in sorted(creator_list, key=lambda x: x[0]):
        print(f"{name:52s}|{namespace:^9s}|{version:^7s}|")
    print("-" * 71)

    # Load / free local plugin creators
    handle = plugin_registry.load_library(str(plugin_file))  # plugin_registry is invalid after this call
    trt.get_plugin_registry().deregister_library(handle)
    handle = trt.get_plugin_registry().load_library(str(plugin_file))  # Load again for later use

    # Geet / register / deregister a plugin creator
    plugin_creator = plugin_registry.get_creator(plugin_name, "1", "")
    trt.get_plugin_registry().register_creator(plugin_creator)
    trt.get_plugin_registry().deregister_creator(plugin_creator)
    trt.get_plugin_registry().register_creator(plugin_creator)  # Load again for later use

    check_api_coverage(plugin_creator)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # plugin_creator._pybind11_conduit_v1_()
    # plugin_field_collection._pybind11_conduit_v1_()
    print(f"{plugin_creator.api_language = }")
    print(f"{plugin_creator.interface_info = }")
    print(f"{plugin_creator.interface_info.kind = }")
    print(f"{plugin_creator.interface_info.major = }")
    print(f"{plugin_creator.interface_info.minor = }")
    print(f"{plugin_creator.name = }")
    print(f"{plugin_creator.plugin_namespace = }")
    print(f"{plugin_creator.plugin_version = }")
    print(f"{plugin_creator.field_names = }")  # Its member can be got but not listed in dir()

    # Print the necessary parameters for creating the plugin
    for i, plugin_field in enumerate(plugin_creator.field_names):
        print(f"{i:2d}->{plugin_field.name},{plugin_field.type},{plugin_field.size},{plugin_field.data}")

    # Feed the PluginCreator with parameters
    plugin_field_collection = trt.PluginFieldCollection()
    plugin_field = trt.PluginField("scalar", np.array([1.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    # tensorrt.PluginFieldType: BF16, CHAR, DIMS, FLOAT16, FLOAT32, FLOAT64, FP4, FP8, INT16, INT32, INT4, INT64, INT8, UNKNOWN
    plugin_field_collection.append(plugin_field)  # Use like a list
    #plugin_field_collection.insert(1,plugin_field)
    #plugin_field_collection.extend([plugin_field])
    #plugin_field_collection.clear()
    #plugin_field_collection.pop(1)

    # Create a plugin from plugin creator
    # tensorrt.TensorRTPhase: BUILD (creating the plugin while building the engine), RUNTIME (creating the plugin while deserializing the engine at runtime, see `trt.TensorRTPhase.RUNTIME`)
    plugin = plugin_creator.create_plugin(plugin_creator.name, plugin_field_collection, trt.TensorRTPhase.BUILD)

    check_api_coverage(plugin)  # Sanity check, unnecessary in normal workflow
    # APIs for plugin life cycle, not used in this example:
    # plugin.clone()
    # plugin.destroy()
    print(f"{plugin.api_language = }")
    print(f"{plugin.interface_info = }")

    interface_core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)

    check_api_coverage(interface_core)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # interface_core._pybind11_conduit_v1_()
    print(f"{interface_core.api_language = }")
    print(f"{interface_core.interface_info = }")
    print(f"{interface_core.interface_info.kind = }")
    print(f"{interface_core.interface_info.major = }")
    print(f"{interface_core.interface_info.minor = }")
    print(f"{interface_core.plugin_name = }")
    print(f"{interface_core.plugin_namespace = }")
    print(f"{interface_core.plugin_version = }")

    interface_build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)

    check_api_coverage(interface_build)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # interface_build._pybind11_conduit_v1_()
    # interface_build.configure_plugin()
    # interface_build.format_combination_limit()
    # interface_build.get_aliased_input()
    # interface_build.get_output_data_types()
    # interface_build.get_output_shapes()
    # interface_build.get_valid_tactics()
    # interface_build.get_workspace_size()
    # interface_build.supports_format_combination()
    print(f"{interface_build.DEFAULT_FORMAT_COMBINATION_LIMIT = }")
    print(f"{interface_build.format_combination_limit = }")
    print(f"{interface_build.interface_info = }")
    print(f"{interface_build.interface_info.kind = }")
    print(f"{interface_build.interface_info.major = }")
    print(f"{interface_build.interface_info.minor = }")
    print(f"{interface_build.metadata_string = }")
    print(f"{interface_build.num_outputs = }")
    print(f"{interface_build.timing_cache_id = }")

    interface_runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)

    check_api_coverage(interface_runtime)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # interface_runtime._pybind11_conduit_v1_()
    # interface_runtime.get_fields_to_serialize()
    # interface_runtime.attach_to_context()
    # interface_runtime.enqueue()
    # interface_runtime.get_fields_to_serialize()
    # interface_runtime.on_shape_change()
    # interface_runtime.set_tactic()
    print(f"{interface_runtime.api_language = }")
    print(f"{interface_runtime.interface_info = }")
    print(f"{interface_runtime.interface_info.kind = }")
    print(f"{interface_runtime.interface_info.major = }")
    print(f"{interface_runtime.interface_info.minor = }")

    # Other classes
    plugin_tensor_desc = trt.PluginTensorDesc()

    check_api_coverage(plugin_tensor_desc)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # plugin_tensor_desc._pybind11_conduit_v1_()
    print(f"{plugin_tensor_desc.dims = }")
    print(f"{plugin_tensor_desc.format = }")
    print(f"{plugin_tensor_desc.scale = }")
    print(f"{plugin_tensor_desc.type = }")

    dynamic_plugin_tensor_desc = trt.DynamicPluginTensorDesc()

    check_api_coverage(dynamic_plugin_tensor_desc)  # Sanity check, unnecessary in normal workflow
    # APIs not used in this example:
    # dynamic_plugin_tensor_desc._pybind11_conduit_v1_()
    print(f"{dynamic_plugin_tensor_desc.desc = }")
    print(f"{dynamic_plugin_tensor_desc.min = }")
    print(f"{dynamic_plugin_tensor_desc.opt = }")
    print(f"{dynamic_plugin_tensor_desc.max = }")

    # Other APIs related to plugin, not used in this example:
    trt.IDimensionExpr  # This class has no constructor defined
    trt.IPluginResourceContext  # This class has no constructor defined!
    trt.DimsExprs()
    trt.IExprBuilder()
    trt.DimensionOperation
    # trt.DimensionOperation: CEIL_DIV, EQUAL, FLOOR_DIV, LESS, MAX, MIN, PROD, SUB, SUM

    shape = (3, 4, 5)
    tw = TRTWrapperV1(plugin_file_list=[plugin_file])
    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(input_tensor.name, [1, 1, 1], shape, shape)
    pluginLayer = tw.network.add_plugin_v3([input_tensor], [], plugin)
    print(pluginLayer.plugin)  # It is a clone of the input `plugin` argument above

@case_mark
def case_plugin_api_reference():
    # Reference / documentation of plugin-related types, enums and methods that
    # cannot be fully exercised without a complete plugin implementation.
    # Each token is referenced literally so it is discoverable as API coverage.

    # ---------------------------------------------------------------------
    # Plugin creator / plugin interface base classes (abstract, no constructor).
    # They are the base classes a Python plugin author subclasses.
    # ---------------------------------------------------------------------
    trt.IPluginCreatorInterface  # Common base interface for all plugin creators
    trt.IPluginCreatorV3Quick  # Creator interface for the "quick" (V3 quick) plugin family
    trt.IPluginV3OneBuildV2  # Build capability (V2) of an IPluginV3 plugin, adds aliased-I/O support
    trt.IPluginV3QuickAOTBuild  # Build capability of a quick plugin that emits Ahead-Of-Time compiled kernels
    trt.IPluginV3QuickBuild  # Build capability of a quick (JIT) plugin
    trt.IPluginV3QuickCore  # Core capability (name / namespace / version) of a quick plugin
    trt.IPluginV3QuickRuntime  # Runtime (enqueue) capability of a quick plugin
    print(f"{trt.IPluginCreatorInterface = }")
    print(f"{trt.IPluginCreatorV3Quick = }")
    print(f"{trt.IPluginV3OneBuildV2 = }")
    print(f"{trt.IPluginV3QuickAOTBuild = }")
    print(f"{trt.IPluginV3QuickBuild = }")
    print(f"{trt.IPluginV3QuickCore = }")
    print(f"{trt.IPluginV3QuickRuntime = }")

    # `trt.PluginFieldCollection_` is the underlying (C++ mangled) type of the
    # container returned/consumed as `trt.PluginFieldCollection`.
    print(f"{trt.PluginFieldCollection_ = }")
    print(f"{(trt.PluginFieldCollection is trt.PluginFieldCollection_) = }")

    # ---------------------------------------------------------------------
    # Enums: printed with their integer values (fully exercised).
    # ---------------------------------------------------------------------
    # trt.PluginArgDataType: data type of an argument passed to a quick plugin kernel.
    print_enumerated_members(trt.PluginArgDataType)
    trt.PluginArgDataType.INT8  # 8-bit integer argument
    trt.PluginArgDataType.INT16  # 16-bit integer argument
    trt.PluginArgDataType.INT32  # 32-bit integer argument

    # trt.PluginArgType: category of a quick-plugin argument.
    print_enumerated_members(trt.PluginArgType)
    trt.PluginArgType.INT  # Integer-typed argument

    # trt.PluginCreatorVersion: which creator ABI/version a creator implements.
    print_enumerated_members(trt.PluginCreatorVersion)
    trt.PluginCreatorVersion.V1  # Native (C++) creator version 1
    trt.PluginCreatorVersion.V1_PYTHON  # Python-implemented creator version 1

    # trt.QuickPluginCreationRequest: how TensorRT should create a quick plugin (JIT vs AOT).
    print_enumerated_members(trt.QuickPluginCreationRequest)
    trt.QuickPluginCreationRequest.UNKNOWN  # No preference specified
    trt.QuickPluginCreationRequest.PREFER_JIT  # Prefer Just-In-Time, fall back to AOT
    trt.QuickPluginCreationRequest.PREFER_AOT  # Prefer Ahead-Of-Time, fall back to JIT
    trt.QuickPluginCreationRequest.STRICT_JIT  # Require JIT, fail otherwise
    trt.QuickPluginCreationRequest.STRICT_AOT  # Require AOT, fail otherwise

    # trt.PluginFieldType: type of a `trt.PluginField` (already used above with FLOAT32).
    print_enumerated_members(trt.PluginFieldType)
    trt.PluginFieldType.DIMS  # Field carrying a dimension (shape) value
    trt.PluginFieldType.UNKNOWN  # Field of unspecified type

    # ---------------------------------------------------------------------
    # trt.KernelLaunchParams: CUDA launch configuration used by AOT quick plugins.
    # It has no public constructor (created by TensorRT), so its attributes are
    # referenced here by name only:
    #   trt.KernelLaunchParams.grid_x, grid_y, grid_z  -> CUDA grid dimensions
    #   trt.KernelLaunchParams.block_x, block_y, block_z -> CUDA block dimensions
    # ---------------------------------------------------------------------
    trt.KernelLaunchParams  # No constructor defined; TensorRT populates it internally
    for attr in ["grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"]:
        assert hasattr(trt.KernelLaunchParams, attr), attr
        print(f"trt.KernelLaunchParams.{attr} = {getattr(trt.KernelLaunchParams, attr)}")

    # ---------------------------------------------------------------------
    # Methods referenced by name (require a live context / full plugin to call):
    # ---------------------------------------------------------------------
    trt.IPluginRegistry.acquire_plugin_resource  # Acquire a shared plugin resource from the registry
    trt.IPluginV2Ext.detach_from_context  # Detach a (deprecated) V2Ext plugin from its execution context
    trt.IPluginV2Ext.get_output_data_type  # Query output data type of a (deprecated) V2Ext plugin
    print(f"{trt.IPluginRegistry.acquire_plugin_resource = }")
    print(f"{trt.IPluginV2Ext.detach_from_context = }")
    print(f"{trt.IPluginV2Ext.get_output_data_type = }")

if __name__ == "__main__":

    # List APIs for plugin v3
    case_plugin_v3()

    # Reference of plugin API surface (types / enums / methods not otherwise exercised)
    case_plugin_api_reference()

    print("Finish")
