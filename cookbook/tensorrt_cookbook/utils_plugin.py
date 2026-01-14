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
from typing import List

import tensorrt as trt
import tensorrt as trt

class DummyBasePlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):

    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "DummyBasePlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        return

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def clone(self) -> trt.IPluginV3:
        cloned_plugin = DummyBasePlugin()
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

class DummyBasePluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "DummyBasePlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        return DummyBasePlugin()

class DummyPluginFactory:

    @staticmethod
    def build(layer_dict: dict, random_name: str = "0"):

        random_plugin_name = f"DummyPlugin-{random_name}"
        random_plugin_creator_name = f"DummyPluginCreator-{random_name}"
        input_tensor_data_type_list = [layer_dict["input_tensor"][i]["data_type"] for i in range(layer_dict["num_output"])]  # TODO: fix this
        output_tensor_data_type_list = [layer_dict["output_tensor"][i]["data_type"] for i in range(layer_dict["num_output"])]  # TODO: fix this
        all_tensor_data_type_list = input_tensor_data_type_list + output_tensor_data_type_list
        output_tensor_shape_list = [layer_dict["output_tensor"][i]["shape"] for i in range(layer_dict["num_output"])]  # TODO: fix this

        class DummyPlugin(DummyBasePlugin):

            def __init__(self):
                super().__init__()
                self.plugin_name = random_plugin_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                return

            def clone(self) -> trt.IPluginV3:
                cloned_plugin = DummyPlugin()
                cloned_plugin.__dict__.update(self.__dict__)
                return cloned_plugin

            def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
                return output_tensor_data_type_list

            def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
                return output_tensor_shape_list

            def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
                desc = in_out[pos].desc
                data_type = desc.type
                format = desc.format  # We assume formats of all input / output tensors are trt.TensorFormat.LINEAR
                return data_type == all_tensor_data_type_list[pos] and (format == trt.TensorFormat.LINEAR or True)

        class DummyPluginCreator(DummyBasePluginCreator):

            def __init__(self):
                super().__init__()
                self.name = random_plugin_creator_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.field_names = trt.PluginFieldCollection([])
                return

            def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
                return DummyPlugin()

        return DummyPluginCreator()
