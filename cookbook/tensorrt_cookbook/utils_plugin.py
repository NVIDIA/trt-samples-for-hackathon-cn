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
"""
import dataclasses
from typing import Optional, Dict
import numpy as np
from .utils_function import datatype_np_to_trtpluginfield

_tensorrt_cookbook_plugin_info_dict = dict()
_tensorrt_cookbook_temporary_plugin_info_dict = dict()

@dataclasses.dataclass
class PluginInfo:
    layer_name  : str
    name        : str
    version     : str
    namespace   : str
    field_collection : Optional[Dict[str, 'trt.PluginFieldCollection']] = None

def get_plugin(plugin_info_dict: dict, b_is_plugin_v2: bool = False):
    '''
    An example of plugin_info_dict:

    {
        "AddScalarPluginLayer":  # Layer name
        {
            "name": "AddScalar",  # Plugin name, version, namespace
            "version": "1",
            "namespace": "",
            "argument_dict":  # Arguments for the plugin
            {
                "scalar": np.array([1.0], dtype=np.float32)
            },
            "number_input_tensor": 1,  # Number of input tensors, used only in pluginv3
            "number_input_shape_tensor": 0,  # Number of input shape tensors, used only in pluginv3
        },
    }
    '''
    plugin_creator = trt.get_plugin_registry().get_creator(plugin_info_dict["name"], plugin_info_dict["version"], plugin_info_dict["namespace"])
    if plugin_creator is None:
        return None
    field_list = []
    for key, value in plugin_info_dict["argument_dict"].items():
        field_list.append(trt.PluginField(key, value, datatype_np_to_trtpluginfield(value.dtype)))
    field_collection = trt.PluginFieldCollection(field_list)
    if b_is_plugin_v2:  # Plugin V2, deprecated
        plugin = plugin_creator.create_plugin(plugin_info_dict["name"], field_collection)
    else:  # Plugin V3
        plugin = plugin_creator.create_plugin(plugin_info_dict["name"], field_collection, trt.TensorRTPhase.BUILD)
    _tensorrt_cookbook_plugin_info_dict[plugin_info_dict["name"]] = plugin
    return plugin

########################################################################################################################
import threading

_tensorrt_cookbook_threading_lock = threading.Lock()

_tensorrt_cookbook_original_add_plugin_v3 = trt.INetworkDefinition.add_plugin_v3

def _tensorrt_cookbook_add_plugin_v3(self, input_tensors: List[trt.ITensor], input_shape_tensors: List[trt.ITensor], plugin: trt.IPluginV3):
    layer = _tensorrt_cookbook_original_add_plugin_v3(self, input_tensors, input_shape_tensors, plugin)
    layer_name = layer.name
    with _tensorrt_cookbook_threading_lock:
        if "temporary_plugin_layer_name" in _tensorrt_cookbook_temporary_plugin_info_dict:
            _tensorrt_cookbook_plugin_info_dict[layer_name] = _tensorrt_cookbook_temporary_plugin_info_dict["temporary_plugin_layer_name"]
        else:
            assert False, f"Cannot find plugin_info_dict for layer: {layer_name}"
    return layer

trt.INetwork.add_plugin_v3 = _tensorrt_cookbook_add_plugin_v3

_tensorrt_cookbook_original_add_plugin_v2 = trt.INetworkDefinition.add_plugin_v2

def _tensorrt_cookbook_add_plugin_v2(self, input_tensors: List[trt.ITensor], plugin: trt.IPluginV2):
    layer = _tensorrt_cookbook_original_add_plugin_v2(self, input_tensors, plugin)
    layer_name = layer.name
    with _tensorrt_cookbook_threading_lock:
        if "temporary_plugin_layer_name" in _tensorrt_cookbook_temporary_plugin_info_dict:
            temporary_dict = _tensorrt_cookbook_temporary_plugin_info_dict.pop("temporary_plugin_layer_name")
            _tensorrt_cookbook_plugin_info_dict[layer_name] = temporary_dict
        else:
            assert False, f"Cannot find plugin_info_dict for layer: {layer_name}"
    return layer

trt.INetwork.add_plugin_v2 = _tensorrt_cookbook_add_plugin_v2

_tensorrt_cookbook_original_get_creator = trt.IPluginRegistry.get_creator

def _tensorrt_cookbook_get_creator(self, name, version="1", namespace=""):
    creator = _tensorrt_cookbook_original_get_creator(self, name, version, namespace)
    if creator is not None:
        assert len(_tensorrt_cookbook_temporary_plugin_info_dict) == 0, "Temporary plugin info dict is not empty."
        _tensorrt_cookbook_temporary_plugin_info_dict["temporary_plugin_layer_name"] = {}

        creator._birth_cert = PluginBirthCert(name, version, namespace)
    return creator

trt.IPluginRegistry.get_creator = _tensorrt_cookbook_get_creator

_tensorrt_cookbook_original_create_plugin = trt.IPluginCreator.create_plugin

def _tensorrt_cookbook_create_plugin(self, name, field_collection, phase=None):
    # 1. 真身调用
    if phase is None:
        plugin = _tensorrt_cookbook_original_create_plugin(self, name, field_collection)
    else:
        plugin = _tensorrt_cookbook_original_create_plugin(self, name, field_collection, phase)

    # 2. 把出生证补全，并挂到 plugin 上
    with _tensorrt_cookbook_threading_lock:
        cert = getattr(self, '_birth_cert', None)
        if cert and cert.name == name:          # 名字对得上才认
            cert.field_collection = field_collection
            _cert_map[plugin] = cert
    return plugin

trt.IPluginCreator.create_plugin = _tensorrt_cookbook_create_plugin

#===============================

_original_create_plugin = trt.IPluginCreator.create_plugin

def _tensorrt_cookbook_create_plugin(self, name, field_collection, phase=None):
    if phase is None:  # Plugin V2, deprecated
        plugin = _original_create_plugin(self, name, field_collection)
    else:  # Plugin V3
        plugin = _original_create_plugin(self, name, field_collection, phase)

    with _tensorrt_cookbook_threading_lock:
        {
            "AddScalarPluginLayer":  # Layer name
            {
                "name": "AddScalar",  # Plugin name, version, namespace
                "version": "1",
                "namespace": "",
                "argument_dict":  # Arguments for the plugin
                {
                    "scalar": np.array([1.0], dtype=np.float32)
                },
                "number_input_tensor": 1,  # Number of input tensors, used only in pluginv3
                "number_input_shape_tensor": 0,  # Number of input shape tensors, used only in pluginv3
            },
        }
        _tensorrt_cookbook_plugin_info_dict[self.name] = plugin
    return plugin

# 3. 猴子补丁
trt.IPluginCreator.create_plugin = _tensorrt_cookbook_create_plugin

########################################################################################################################
"""

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
