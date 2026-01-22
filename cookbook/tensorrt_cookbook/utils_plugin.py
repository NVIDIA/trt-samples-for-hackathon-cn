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

from typing import List
import json
import tensorrt as trt
import threading
import numpy as np
from .utils_function import datatype_trt_pluginfield_to_np, datatype_np_to_trt_pluginfield

_tensorrt_cookbook_threading_lock = threading.Lock()

_tensorrt_cookbook_enable_plugin_hook = False
_tensorrt_cookbook_plugin_info_dict = dict()

temporary_plugin_layer_name = "TPLN"

def get_plugin(user_plugin_info: dict):
    """
    Standard format of a `plugin_info` (both user's and internal usage):
    {
        name:                       str = ""
        version:                    str = "1"
        namespace:                  str = ""
        argument_dict:              Optional[Dict[str, np.array]] = {}
        number_input_tensor:        int = 1                             # Number of input tensors, used only in pluginv3
        number_input_shape_tensor:  int = 0                             # Number of input shape tensors, used only in pluginv3
        plugin_api_version:         int = "3"                           # 3 for Plugin V3, 2 for Plugin V2 (deprecated)
    }

    In a network, we may have more than one plugins, so we use `plugin_info_dict` to manage all plugins:
    {
        layer_name_0: plugin_info_0,
        layer_name_1: plugin_info_1,
        ...
    }
    """
    plugin_creator = trt.get_plugin_registry().get_creator(user_plugin_info["name"], user_plugin_info["version"], user_plugin_info["namespace"])
    if plugin_creator is None:
        return None
    field_list = []
    for key, value in user_plugin_info["argument_dict"].items():
        field_list.append(trt.PluginField(key, value, datatype_np_to_trt_pluginfield(value.dtype)))
    field_collection = trt.PluginFieldCollection(field_list)
    if user_plugin_info["plugin_api_version"] == "3":  # Plugin V3
        plugin = plugin_creator.create_plugin(user_plugin_info["name"], field_collection, trt.TensorRTPhase.BUILD)
    else:  # Plugin V2, deprecated
        plugin = plugin_creator.create_plugin(user_plugin_info["name"], field_collection)
    return plugin

_tensorrt_cookbook_original_add_plugin_v3 = trt.INetworkDefinition.add_plugin_v3

def _tensorrt_cookbook_add_plugin_v3(self, input_tensors: List[trt.ITensor], input_shape_tensors: List[trt.ITensor], plugin: trt.IPluginV3):
    layer = _tensorrt_cookbook_original_add_plugin_v3(self, input_tensors, input_shape_tensors, plugin)
    layer_name = layer.name
    with _tensorrt_cookbook_threading_lock:
        internal_plugin_info = _tensorrt_cookbook_plugin_info_dict.pop(temporary_plugin_layer_name, None)
        assert internal_plugin_info is not None, f"Cannot find internal_plugin_info for layer: {layer_name}"
        internal_plugin_info["number_input_tensor"] = len(input_tensors)
        internal_plugin_info["number_input_shape_tensor"] = len(input_shape_tensors)
        internal_plugin_info["plugin_api_version"] = "3"
        _tensorrt_cookbook_plugin_info_dict[layer_name] = internal_plugin_info
    return layer

_tensorrt_cookbook_original_add_plugin_v2 = trt.INetworkDefinition.add_plugin_v2

def _tensorrt_cookbook_add_plugin_v2(self, input_tensors: List[trt.ITensor], plugin: trt.IPluginV2):
    layer = _tensorrt_cookbook_original_add_plugin_v2(self, input_tensors, plugin)
    layer_name = layer.name  # TODO: how can we find the plugin if the layer name is changed?
    with _tensorrt_cookbook_threading_lock:
        internal_plugin_info = _tensorrt_cookbook_plugin_info_dict.pop(temporary_plugin_layer_name, None)
        assert internal_plugin_info is not None, f"Cannot find internal_plugin_info for layer: {layer_name}"
        internal_plugin_info["number_input_tensor"] = len(input_tensors)
        internal_plugin_info["number_input_shape_tensor"] = 0
        internal_plugin_info["plugin_api_version"] = "2"
        _tensorrt_cookbook_plugin_info_dict[layer_name] = internal_plugin_info
    return layer

_tensorrt_cookbook_original_create_plugin_V3One = trt.IPluginCreatorV3One.create_plugin  # Plugin V3
_tensorrt_cookbook_original_create_plugin = trt.IPluginCreator.create_plugin  # Plugin V2, deprecated

def _tensorrt_cookbook_create_plugin(self, name, field_collection, phase=None):
    with _tensorrt_cookbook_threading_lock:
        internal_plugin_info = _tensorrt_cookbook_plugin_info_dict.get(temporary_plugin_layer_name, None)
        assert internal_plugin_info is not None, f"Cannot find internal_plugin_info for plugin: {name}"
        argument_dict = {}
        for field in field_collection:
            argument_dict[field.name] = np.array(field.data, dtype=datatype_trt_pluginfield_to_np(field.type))
        internal_plugin_info["argument_dict"] = argument_dict

    # TODO: Use a better way to distinguish Plugin V3 and V2, for example, `"IPluginCreatorV3One" in str(type(self))`
    if phase is not None:  # Plugin V3
        plugin = _tensorrt_cookbook_original_create_plugin_V3One(self, name, field_collection, phase)
    else:  # Plugin V2
        plugin = _tensorrt_cookbook_original_create_plugin(self, name, field_collection)

    # Rehook the method after creating the plugin
    trt.IPluginCreatorV3One.create_plugin = _tensorrt_cookbook_original_create_plugin_V3One
    trt.IPluginCreator.create_plugin = _tensorrt_cookbook_original_create_plugin
    return plugin

_tensorrt_cookbook_original_get_creator = trt.IPluginRegistry.get_creator

def _tensorrt_cookbook_get_creator(self, name, version="1", namespace=""):
    creator = _tensorrt_cookbook_original_get_creator(self, name, version, namespace)
    if creator is None:
        return None
    internal_plugin_info = dict(name=name, version=version, namespace=namespace)  # New item of `_tensorrt_cookbook_plugin_info_dict` is created here
    # Hook the method after getting the creator
    trt.IPluginCreatorV3One.create_plugin = _tensorrt_cookbook_create_plugin
    trt.IPluginCreator.create_plugin = _tensorrt_cookbook_create_plugin
    _tensorrt_cookbook_plugin_info_dict[temporary_plugin_layer_name] = internal_plugin_info
    return creator

_tensorrt_cookbook_original_plugin_v3_layer_name_setter = trt.IPluginV3Layer.name.fset
_tensorrt_cookbook_original_plugin_v2_layer_name_setter = trt.IPluginV2Layer.name.fset

def hooked_name_setter(self, new_name):
    """
    Edit _tensorrt_cookbook_plugin_info_dict when the name of plugin layer is changed.
    """
    old_name = self.name
    with _tensorrt_cookbook_threading_lock:
        internal_plugin_info = _tensorrt_cookbook_plugin_info_dict.pop(old_name, None)
        assert internal_plugin_info is not None, f"Cannot find internal_plugin_info for layer: {old_name}"
        _tensorrt_cookbook_plugin_info_dict[new_name] = internal_plugin_info
    if isinstance(self, trt.IPluginV3Layer):  # Plugin V3
        _tensorrt_cookbook_original_plugin_v3_layer_name_setter(self, new_name)
    else:  # Plugin V2, deprecated
        _tensorrt_cookbook_original_plugin_v2_layer_name_setter(self, new_name)
    return

# With plugin hooks, plugin layer information can be recorded without explicitly using plugin_info_dict and `get_plugin`.
# Users can just use their own worklow to build network with plugins, and the cookbook is able to get information for later serialization.
def enable_plugin_hook():
    global _tensorrt_cookbook_enable_plugin_hook
    _tensorrt_cookbook_enable_plugin_hook = True
    trt.INetworkDefinition.add_plugin_v3 = _tensorrt_cookbook_add_plugin_v3
    trt.INetworkDefinition.add_plugin_v2 = _tensorrt_cookbook_add_plugin_v2
    trt.IPluginRegistry.get_creator = _tensorrt_cookbook_get_creator
    trt.IPluginV3Layer.name = property(trt.IPluginV3Layer.name.fget, hooked_name_setter)
    trt.IPluginV2Layer.name = property(trt.IPluginV2Layer.name.fget, hooked_name_setter)

def disable_plugin_hook():
    global _tensorrt_cookbook_enable_plugin_hook
    _tensorrt_cookbook_enable_plugin_hook = False
    trt.INetworkDefinition.add_plugin_v3 = _tensorrt_cookbook_original_add_plugin_v3
    trt.INetworkDefinition.add_plugin_v2 = _tensorrt_cookbook_original_add_plugin_v2
    trt.IPluginRegistry.get_creator = _tensorrt_cookbook_original_get_creator
    trt.IPluginV3Layer.name = property(trt.IPluginV3Layer.name.fget, _tensorrt_cookbook_original_plugin_v3_layer_name_setter)
    trt.IPluginV2Layer.name = property(trt.IPluginV2Layer.name.fget, _tensorrt_cookbook_original_plugin_v2_layer_name_setter)

class DummyBasePluginV3(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "DummyBasePluginV3"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        return

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def clone(self) -> trt.IPluginV3:
        cloned_plugin = DummyBasePluginV3()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return input_types

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        return inputs

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        return True

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self: trt.IPluginV3, tactic: int) -> None:
        return None

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> ModuleNotFoundError:
        return None

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([])

class DummyBasePluginV3Creator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "DummyBasePluginV3"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        return DummyBasePluginV3()

class DummyBasePluginV2(trt.IPluginV2DynamicExt):

    def __init__(self):
        super().__init__()
        self.plugin_type = "DummyBasePluginV2"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        return

    def initialize(self) -> int:
        return 0

    def terminate(self) -> None:
        return

    def get_serialization_size(self) -> int:
        return 0

    def serialize(self) -> bytes:
        return json.dumps({})

    def destroy(self) -> None:
        return

    def get_output_datatype(self, output_index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned_plugin = DummyBasePluginV2(0.0)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        return True

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_workspace_size(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        return

class DummyBasePluginV2Creator(trt.IPluginCreator):

    def __init__(self):
        super().__init__()
        self.name = "DummyBasePluginV2"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name, plugin_field_collection):
        return DummyBasePluginV2()

    def deserialize_plugin(self, name, data):
        return DummyBasePluginV2()

class DummyPluginFactory:

    @staticmethod
    def build(layer_dict: dict, tensor_dict: dict):

        plugin_name = layer_dict["name"]
        plugin_creator_name = layer_dict["name"]
        input_tensor_data_type_list = [trt.DataType(tensor_dict[name]["dtype"]) for name in layer_dict["input_tensor_name_list"]]
        output_tensor_data_type_list = [trt.DataType(tensor_dict[name]["dtype"]) for name in layer_dict["output_tensor_name_list"]]
        all_tensor_data_type_list = input_tensor_data_type_list + output_tensor_data_type_list
        itsl = [tensor_dict[name]["shape"] for name in layer_dict["input_tensor_name_list"]]  # Short ffor "input_tensor_shape_list"
        otsl = [tensor_dict[name]["shape"] for name in layer_dict["output_tensor_name_list"]]  # Short for "output_tensor_shape_list"

        def _compute_output_shapes(
            itsl: List[list],
            otsl: List[list],
            inputs: List[trt.DimsExprs],
            shape_inputs: List[trt.DimsExprs],
            expr_builder: trt.IExprBuilder,
        ):
            if len(itsl) == len(otsl) and all([its == ots for its, ots in zip(itsl, otsl)]):
                # Case: The shape of output tensors are one-one-map to the input tensors'
                # [[a_00,a_01,a_02,...],[a_10,a_11,a_12,...],...] -> [[a_00,a_01,a_02,...],[a_10,a_11,a_12,...],...]
                return inputs

            elif any([shape[0] == -1 for shape in itsl]) and any([shape[0] == -1 for shape in otsl]) and \
                all([-1 not in shape[1:] for shape in itsl]) and all([-1 not in shape[1:] for shape in otsl]):
                # Case: At least one input and one output tensors have the dynamic first dimension (-1)
                # [[-1,a_01,a_02,...],[a_10,a_11,a_12,...],...] -> [[-1,b_01,b_02,...],[b_10,b_11,b_12,...],...]
                n1_i = 0
                for i in range(len(itsl)):
                    if itsl[i][0] == -1:
                        n1_i = i
                        break
                output_dims_list = []
                for i in range(len(otsl)):
                    output_dims = trt.DimsExprs(len(otsl[i]))
                    for j in range(len(otsl[i])):
                        output_dims[j] = expr_builder.constant(otsl[i][j])
                    if otsl[i][0] == -1:
                        output_dims[0] = inputs[n1_i][0]
                    output_dims_list.append(output_dims)

                return output_dims_list

            elif any([shape[0:2] == [-1, -1] for shape in itsl]) and any([shape[0:2] == [-1, -1] for shape in otsl]) and \
                all([-1 not in shape[2:] for shape in itsl]) and all([-1 not in shape[2:] for shape in otsl]):
                # Case: At least one input and one output tensors have the dynamic first two dimension (-1,-1)
                # [[-1,-1,a_02,...],[-1,a_11,a_12,...],[a_20,a_21,a_22,...],...] -> [[-1,-1,b_02,...],[-1,b_11,b_12,...],[b_20,b_21,b_22,...],...]
                n1_i = 0
                for i in range(len(itsl)):
                    if itsl[i][0:2] == [-1, -1]:
                        n1_i = i
                        break
                output_dims_list = []
                for i in range(len(otsl)):
                    output_dims = trt.DimsExprs(len(otsl[i]))
                    for j in range(len(otsl[i])):
                        output_dims[j] = expr_builder.constant(otsl[i][j])
                    if otsl[i][0:2] == [-1, -1]:
                        output_dims[0] = inputs[n1_i][0]
                        output_dims[1] = inputs[n1_i][1]
                    if otsl[i][0] == -1:
                        output_dims[0] = inputs[n1_i][0]
                    output_dims_list.append(output_dims)

                return output_dims_list

            elif any([shape[0:3] == [-1, -1, -1] for shape in itsl]) and any([shape[0:3] == [-1, -1, -1] for shape in otsl]) and \
                all([-1 not in shape[3:] for shape in itsl]) and all([-1 not in shape[3:] for shape in otsl]):
                # Case: At least one input and one output tensors have the dynamic first three dimension (-1,-1)
                # input : [[-1,-1,-1,...],[-1,-1,a_12,...],[-1,a_21,a_22,...],[a_30,a_31,a_32,...],...]
                # output: [[-1,-1,-1,...],[-1,-1,b_12,...],[-1,b_21,b_22,...],[b_30,b_31,b_32,...],...]
                n1_i = 0
                for i in range(len(itsl)):
                    if itsl[i][0:3] == [-1, -1, -1]:
                        n1_i = i
                        break
                output_dims_list = []
                for i in range(len(otsl)):
                    output_dims = trt.DimsExprs(len(otsl[i]))
                    for j in range(len(otsl[i])):
                        output_dims[j] = expr_builder.constant(otsl[i][j])
                    if otsl[i][0:3] == [-1, -1, -1]:
                        output_dims[0] = inputs[n1_i][0]
                        output_dims[1] = inputs[n1_i][1]
                        output_dims[2] = inputs[n1_i][2]
                    if otsl[i][0:2] == [-1, -1]:
                        output_dims[0] = inputs[n1_i][0]
                        output_dims[1] = inputs[n1_i][1]
                    if otsl[i][0] == -1:
                        output_dims[0] = inputs[n1_i][0]
                    output_dims_list.append(output_dims)

                return output_dims_list

            elif any([shape[-1] == -1 for shape in itsl]) and any([shape[-1] == -1 for shape in otsl]) and \
                all([-1 not in shape[:-1] for shape in itsl]) and all([-1 not in shape[:-1] for shape in otsl]):
                # Case: At least one input and one output tensors have the dynamic first dimension (-1)
                # [[a_00,a_01,a_02,...,-1],[a_10,a_11,a_12,...],...] -> [[b_00,b_01,b_02,...,-1],[b_10,b_11,b_12,...],...]
                n1_i = 0
                for i in range(len(itsl)):
                    if itsl[i][-1] == -1:
                        n1_i = i
                        break
                output_dims_list = []
                for i in range(len(otsl)):
                    output_dims = trt.DimsExprs(len(otsl[i]))
                    for j in range(len(otsl[i])):
                        output_dims[j] = expr_builder.constant(otsl[i][j])
                    if otsl[i][-1] == -1:
                        output_dims[-1] = inputs[n1_i][-1]
                    output_dims_list.append(output_dims)

                return output_dims_list

            elif any([[shape[0],shape[-1]] == [-1, -1] for shape in itsl]) and any([[shape[0],shape[-1]] == [-1, -1] for shape in otsl]) and \
                all([-1 not in shape[1:-1] for shape in itsl]) and all([-1 not in shape[1:-1] for shape in otsl]):
                # Case: At least one input and one output tensors have the dynamic first two dimension (-1,-1)
                # [[-1,a_01,a_02,...,-1],[a_10,a_11,a_12,...],...] -> [[-1,b_01,b_02,...,-1],[b_10,b_11,b_12,...],...]
                n1_i = 0
                for i in range(len(itsl)):
                    if [itsl[i][0], itsl[i][-1]] == [-1, -1]:
                        n1_i = i
                        break
                output_dims_list = []
                for i in range(len(otsl)):
                    output_dims = trt.DimsExprs(len(otsl[i]))
                    for j in range(len(otsl[i])):
                        output_dims[j] = expr_builder.constant(otsl[i][j])
                    if [otsl[i][0], otsl[i][-1]] == [-1, -1]:
                        output_dims[0] = inputs[n1_i][0]
                        output_dims[-1] = inputs[n1_i][-1]
                    if otsl[i][0] == -1:
                        output_dims[0] = inputs[n1_i][0]
                    output_dims_list.append(output_dims)

                return output_dims_list

            else:
                # Possible situation but not considered:
                # Matmul-like: [a,-1,c],[a,c,-2] -> [a,-1,-2]
                # Reduce-like: [a,b,-1] -> [a,b,1,], or [a,b,-1] -> [a,b]
                # Reshape-like: [a,-1,c] -> [a,-1,e,f]
                # Squeeze/Unsqueeze-like: [a,b,-1] -> [a,b,1,-1], or [a,b,1,-1] -> [a,b,-1]
                # Sum-like: [a,b], [a,c] -> [a,b+c]
                # Tail -1 like: [a,-1], [a,-1] -> [a,-1]
                # Transpose-like: [a,b,-2] -> [a,-2,b], or [a,-1,-2] -> [a,-2,-1]
                return inputs

        class DummyPluginV3(DummyBasePluginV3):

            def __init__(self):
                super().__init__()
                self.plugin_name = plugin_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.num_outputs = len(output_tensor_data_type_list)
                return

            def clone(self) -> trt.IPluginV3:
                cloned_plugin = DummyPluginV3()
                cloned_plugin.__dict__.update(self.__dict__)
                return cloned_plugin

            def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
                return output_tensor_data_type_list

            def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
                return _compute_output_shapes(itsl, otsl, inputs, shape_inputs, expr_builder)

            def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
                desc = in_out[pos].desc
                data_type = desc.type
                format = desc.format  # We assume formats of all input / output tensors are trt.TensorFormat.LINEAR
                return data_type == all_tensor_data_type_list[pos] and (format == trt.TensorFormat.LINEAR or True)

        class DummyPluginV3Creator(DummyBasePluginV3Creator):

            def __init__(self):
                super().__init__()
                self.name = plugin_creator_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.field_names = trt.PluginFieldCollection([])
                return

            def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
                return DummyPluginV3()

        class DummyPluginV2(DummyBasePluginV2):

            def __init__(self):
                super().__init__()
                self.plugin_type = plugin_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.num_outputs = len(output_tensor_data_type_list)
                return

            def get_output_datatype(self, output_index: int, input_types: List[trt.DataType]) -> trt.DataType:
                return output_tensor_data_type_list[output_index]

            def clone(self) -> trt.IPluginV2DynamicExt:
                cloned_plugin = DummyPluginV2()
                cloned_plugin.__dict__.update(self.__dict__)
                return cloned_plugin

            def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
                return _compute_output_shapes(itsl, otsl, inputs, [], expr_builder)[output_index]

            def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
                data_type = in_out[pos].type
                format = in_out[pos].format  # We assume formats of all input / output tensors are trt.TensorFormat.LINEAR
                return data_type == all_tensor_data_type_list[pos] and (format == trt.TensorFormat.LINEAR or True)

        class DummyPluginV2Creator(trt.IPluginCreator):

            def __init__(self):
                trt.IPluginCreator.__init__(self)
                self.name = plugin_creator_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.field_names = trt.PluginFieldCollection([])
                return

            def create_plugin(self, name, plugin_field_collection):
                return DummyPluginV2()

            def deserialize_plugin(self, name, data):
                return DummyPluginV2()

        if trt.LayerType(layer_dict["type"]) == trt.LayerType.PLUGIN_V3:
            return DummyPluginV3Creator()
        else:  # trt.LayerType.IPluginV2Layer:
            return DummyPluginV2Creator()
