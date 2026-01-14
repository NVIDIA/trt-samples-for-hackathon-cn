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
        self.num_outputs = 1
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
    def build(layer_dict: dict, tensor_dict: dict):

        random_plugin_name = layer_dict["name"]
        random_plugin_creator_name = layer_dict["name"]
        input_tensor_data_type_list = [trt.DataType(tensor_dict[name]["dtype"]) for name in layer_dict["input_tensor_name_list"]]
        output_tensor_data_type_list = [trt.DataType(tensor_dict[name]["dtype"]) for name in layer_dict["output_tensor_name_list"]]
        all_tensor_data_type_list = input_tensor_data_type_list + output_tensor_data_type_list
        input_tensor_shape_list = [tensor_dict[name]["shape"] for name in layer_dict["input_tensor_name_list"]]
        output_tensor_shape_list = [tensor_dict[name]["shape"] for name in layer_dict["output_tensor_name_list"]]

        def _compute_output_shapes(
            input_tensor_shape_list: List[list],
            output_tensor_shape_list: List[list],
            inputs: List[trt.DimsExprs],
            shape_inputs: List[trt.DimsExprs],
            expr_builder: trt.IExprBuilder,
        ):

            compute_output_shapes = lambda inputs, expr_builder: inputs  # Default version

            if len(input_tensor_shape_list) == len(output_tensor_shape_list) and all([its == ots for its, ots in zip(input_tensor_shape_list, output_tensor_shape_list)]):
                # Case 0, shapes of all output tensors are the same as the input tensors
                # So we use the exact input tensor shapes as the output tensor shapes
                return lambda inputs, expr_builder: [trt.DimsExprs(inputs[0])]

            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 1: Single input and single output, both with dynamic first dimension (-1)
                # Copy the dynamic first dimension from input, and use static shape for remaining dimensions
                def compute_output_shapes_case1(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Copy the first dimension from input
                    for i in range(1, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])  # Use static values for remaining dimensions
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case1
            elif all(shape[0] == -1 for shape in input_tensor_shape_list) and all(shape[0] == -1 for shape in output_tensor_shape_list):
                # Case 2: All tensors have dynamic first dimension (-1)
                # Copy the dynamic first dimension from first input to all outputs, use static shape for remaining dimensions
                def compute_output_shapes_case2(inputs, expr_builder):
                    result = []
                    for ots in output_tensor_shape_list:
                        output_dims = trt.DimsExprs(len(ots))
                        output_dims[0] = inputs[0][0]  # Copy the first dimension from first input
                        for i in range(1, len(ots)):
                            output_dims[i] = expr_builder.constant(ots[i])  # Use static values for remaining dimensions
                        result.append(output_dims)
                    return result

                compute_output_shapes = compute_output_shapes_case2
            elif len(input_tensor_shape_list) > 1 and len(output_tensor_shape_list) == 1 and \
                (any(len(shape) >= 2 and shape[0] == -1 and shape[1] == -1 for shape in input_tensor_shape_list)) and \
                len(output_tensor_shape_list[0]) >= 2 and output_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][1] == -1:
                # Case 3: Multiple inputs, single output, at least one input has dynamic first two dimensions
                # Output also has dynamic first two dimensions
                # Copy the first two dynamic dimensions from the first input with two dynamic dimensions
                def compute_output_shapes_case3(inputs, expr_builder):
                    # Find the first input with two dynamic dimensions
                    source_input_idx = 0
                    for idx, shape in enumerate(input_tensor_shape_list):
                        if len(shape) >= 2 and shape[0] == -1 and shape[1] == -1:
                            source_input_idx = idx
                            break

                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[source_input_idx][0]  # Copy first dimension
                    output_dims[1] = inputs[source_input_idx][1]  # Copy second dimension
                    for i in range(2, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])  # Use static values for remaining dimensions
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case3
            elif len(input_tensor_shape_list) == 2 and len(output_tensor_shape_list) == 1 and \
                all(len(shape) >= 2 for shape in input_tensor_shape_list) and len(output_tensor_shape_list[0]) >= 2 and \
                input_tensor_shape_list[0][0] == -1 and input_tensor_shape_list[1][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 4: Concat-like operation - two inputs with dynamic batch, output batch is sum of input batches
                # Example: input1 [-1,C,H,W], input2 [-1,C,H,W] -> output [-1,C,H,W] where output_batch = input1_batch + input2_batch
                def compute_output_shapes_case4(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = expr_builder.operation(trt.DimensionOperation.SUM, *inputs[0][0], *inputs[1][0])
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]  # Copy from first input
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case4
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) >= 2 and \
                input_tensor_shape_list[0][0] == -1 and input_tensor_shape_list[0][1] == -1 and \
                output_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][1] != -1:
                # Case 5: Reshape-like operation - dynamic batch and sequence length in, dynamic batch out with fixed other dims
                # Example: input [-1,-1,C] -> output [-1,C',H,W] where output_batch = input_batch
                def compute_output_shapes_case5(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Copy batch dimension
                    for i in range(1, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case5
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 3 and len(output_tensor_shape_list[0]) >= 3 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                all(input_tensor_shape_list[0][i] == -1 for i in range(1, min(3, len(input_tensor_shape_list[0])))) and \
                all(output_tensor_shape_list[0][i] == -1 for i in range(1, min(3, len(output_tensor_shape_list[0])))):
                # Case 6: Multi-dynamic dimensions - first 3 dimensions all dynamic (common in NLP/Transformer)
                # Example: input [-1,-1,-1,C] -> output [-1,-1,-1,C']
                def compute_output_shapes_case6(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # Copy first 3 dynamic dimensions
                    for i in range(min(3, len(output_tensor_shape_list[0]))):
                        if i < len(inputs[0]):
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    # Static dimensions for the rest
                    for i in range(3, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case6
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 3 and len(output_tensor_shape_list[0]) == 2 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 7: Reduce/Pooling operation - reduce spatial dimensions
                # Example: input [-1,C,H,W] -> output [-1,C] (global pooling)
                def compute_output_shapes_case7(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Copy batch dimension
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case7
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) > len(input_tensor_shape_list[0]) and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 8: Expand/Upsample operation - increase dimensions
                # Example: input [-1,C] -> output [-1,C,H,W] (expand to 4D)
                def compute_output_shapes_case8(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # Copy dynamic dimensions from input
                    for i in range(len(input_tensor_shape_list[0])):
                        if input_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    # Fill remaining dimensions with static values
                    for i in range(len(input_tensor_shape_list[0]), len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case8
            elif len(input_tensor_shape_list) == 2 and len(output_tensor_shape_list) == 1 and \
                all(len(shape) >= 2 for shape in input_tensor_shape_list) and \
                input_tensor_shape_list[0][0] == -1 and input_tensor_shape_list[1][0] == -1 and \
                output_tensor_shape_list[0][0] == -1 and len(output_tensor_shape_list[0]) >= 2:
                # Case 9: MatMul-like operation - two inputs with compatible shapes
                # Example: input1 [-1,M,K], input2 [-1,K,N] -> output [-1,M,N]
                def compute_output_shapes_case9(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Copy batch from first input
                    if len(output_tensor_shape_list[0]) >= 2:
                        if output_tensor_shape_list[0][1] == -1 and len(input_tensor_shape_list[0]) >= 2 and input_tensor_shape_list[0][1] == -1:
                            output_dims[1] = inputs[0][1]  # M dimension from first input
                        else:
                            output_dims[1] = expr_builder.constant(output_tensor_shape_list[0][1])
                    if len(output_tensor_shape_list[0]) >= 3:
                        if output_tensor_shape_list[0][2] == -1 and len(input_tensor_shape_list[1]) >= 2 and input_tensor_shape_list[1][-1] == -1:
                            output_dims[2] = inputs[1][-1]  # N dimension from second input
                        else:
                            output_dims[2] = expr_builder.constant(output_tensor_shape_list[0][2])
                    for i in range(3, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case9
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) >= 2 and \
                all(shape[0] == -1 for shape in output_tensor_shape_list) and input_tensor_shape_list[0][0] == -1:
                # Case 10: Split-like operation - one input split into multiple outputs
                # Example: input [-1,C,H,W] -> output1 [-1,C1,H,W], output2 [-1,C2,H,W]
                def compute_output_shapes_case10(inputs, expr_builder):
                    result = []
                    for ots in output_tensor_shape_list:
                        output_dims = trt.DimsExprs(len(ots))
                        for i in range(len(ots)):
                            if ots[i] == -1:
                                if i < len(inputs[0]):
                                    output_dims[i] = inputs[0][i]
                                else:
                                    output_dims[i] = expr_builder.constant(1)
                            else:
                                output_dims[i] = expr_builder.constant(ots[i])
                        result.append(output_dims)
                    return result

                compute_output_shapes = compute_output_shapes_case10
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 4 and len(output_tensor_shape_list[0]) >= 4 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                input_tensor_shape_list[0][-2] == -1 and input_tensor_shape_list[0][-1] == -1 and \
                output_tensor_shape_list[0][-2] == -1 and output_tensor_shape_list[0][-1] == -1:
                # Case 11: Resize/Interpolate operation - dynamic spatial dimensions
                # Example: input [-1,C,-1,-1] -> output [-1,C,-1,-1] with spatial dims scaled
                def compute_output_shapes_case11(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch
                    for i in range(1, len(output_tensor_shape_list[0]) - 2):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    # Last two dimensions (H, W) - could be scaled
                    output_dims[-2] = inputs[0][-2]
                    output_dims[-1] = inputs[0][-1]
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case11
            elif len(input_tensor_shape_list) >= 2 and len(output_tensor_shape_list) == 1 and \
                all(shape[0] == -1 for shape in input_tensor_shape_list) and \
                output_tensor_shape_list[0][0] == -1 and len(output_tensor_shape_list[0]) >= 2:
                # Case 12: Concat along channel dimension - multiple inputs concatenated
                # Example: input1 [-1,C1,H,W], input2 [-1,C2,H,W] -> output [-1,C1+C2,H,W]
                def compute_output_shapes_case12(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch from first input
                    # For channel dimension, check if we need to sum
                    if len(output_tensor_shape_list[0]) >= 2 and output_tensor_shape_list[0][1] == -1:
                        # Sum channel dimensions from all inputs
                        channel_sum = inputs[0][1]
                        for i in range(1, len(inputs)):
                            if len(inputs[i]) >= 2:
                                channel_sum = expr_builder.operation(trt.DimensionOperation.SUM, *channel_sum, *inputs[i][1])
                        output_dims[1] = channel_sum
                    else:
                        output_dims[1] = expr_builder.constant(output_tensor_shape_list[0][1])
                    # Copy remaining dimensions
                    for i in range(2, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case12
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 3 and len(output_tensor_shape_list[0]) >= 3 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                input_tensor_shape_list[0][-2:] != output_tensor_shape_list[0][-2:]:
                # Case 13: Convolution/Deconvolution - spatial dimensions change
                # Example: input [-1,C,H,W] -> output [-1,C',H',W']
                def compute_output_shapes_case13(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1 and i < len(inputs[0]):
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case13
            elif len(input_tensor_shape_list) == 2 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 1 and len(input_tensor_shape_list[1]) == 1 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 14: Gather/Scatter - indexing operation with data and indices
                # Example: data [-1,C,H,W], indices [-1] -> output [-1,H,W] or similar
                def compute_output_shapes_case14(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # Usually batch from data tensor
                    output_dims[0] = inputs[0][0]
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            # Try to infer from input shapes
                            if i < len(inputs[0]):
                                output_dims[i] = inputs[0][i]
                            elif len(inputs) > 1 and i - len(inputs[0]) + len(inputs[1]) < len(inputs[1]):
                                output_dims[i] = inputs[1][i - len(inputs[0]) + len(inputs[1])]
                            else:
                                output_dims[i] = expr_builder.constant(1)
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case14
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) == len(input_tensor_shape_list[0]) - 1 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 15: Squeeze - remove a dimension of size 1
                # Example: input [-1,1,H,W] -> output [-1,H,W]
                def compute_output_shapes_case15(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    j = 0
                    for i in range(len(inputs[0])):
                        if j < len(output_tensor_shape_list[0]) and input_tensor_shape_list[0][i] != 1:
                            if output_tensor_shape_list[0][j] == -1:
                                output_dims[j] = inputs[0][i]
                            else:
                                output_dims[j] = expr_builder.constant(output_tensor_shape_list[0][j])
                            j += 1
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case15
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) == len(input_tensor_shape_list[0]) + 1 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1:
                # Case 16: Unsqueeze - add a dimension of size 1
                # Example: input [-1,C,H] -> output [-1,C,1,H] or [-1,C,H,1]
                def compute_output_shapes_case16(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    j = 0
                    for i in range(len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == 1:
                            output_dims[i] = expr_builder.constant(1)
                        else:
                            if j < len(inputs[0]):
                                if output_tensor_shape_list[0][i] == -1:
                                    output_dims[i] = inputs[0][j]
                                else:
                                    output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                                j += 1
                            else:
                                output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case16
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 3 and len(output_tensor_shape_list[0]) >= 3 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                all(dim != -1 for dim in output_tensor_shape_list[0][1:]):
                # Case 17: Reduce along specific dimensions
                # Example: input [-1,C,H,W] -> output [-1,1,1,1] (reduce all but batch) or [-1,C,1,1] (spatial reduce)
                def compute_output_shapes_case17(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Keep batch
                    for i in range(1, len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case17
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 2 and \
                input_tensor_shape_list[0][0] == -1 and \
                output_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[1][0] == -1:
                # Case 18: TopK - returns values and indices
                # Example: input [-1,C] -> output_values [-1,K], output_indices [-1,K]
                def compute_output_shapes_case18(inputs, expr_builder):
                    result = []
                    for ots in output_tensor_shape_list:
                        output_dims = trt.DimsExprs(len(ots))
                        output_dims[0] = inputs[0][0]  # Batch
                        for i in range(1, len(ots)):
                            if ots[i] == -1:
                                output_dims[i] = inputs[0][i] if i < len(inputs[0]) else expr_builder.constant(1)
                            else:
                                output_dims[i] = expr_builder.constant(ots[i])
                        result.append(output_dims)
                    return result

                compute_output_shapes = compute_output_shapes_case18
            elif len(input_tensor_shape_list) == 3 and len(output_tensor_shape_list) == 1 and \
                all(shape[0] == -1 for shape in input_tensor_shape_list) and output_tensor_shape_list[0][0] == -1:
                # Case 19: Attention mechanism - Q, K, V inputs
                # Example: Q [-1,seq,d], K [-1,seq,d], V [-1,seq,d] -> output [-1,seq,d]
                def compute_output_shapes_case19(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch from Q
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            # Usually follows Q tensor shape
                            output_dims[i] = inputs[0][i] if i < len(inputs[0]) else expr_builder.constant(1)
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case19
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) >= 2 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                len(input_tensor_shape_list[0]) == len(output_tensor_shape_list[0]) and \
                input_tensor_shape_list[0] != output_tensor_shape_list[0]:
                # Case 20: Shuffle/Transpose - permute dimensions with dynamic batch
                # Example: input [-1,C,H,W] -> output [-1,H,W,C] (NCHW to NHWC)
                def compute_output_shapes_case20(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # For transpose, we need to map input dims to output dims
                    # Since we don't know the exact permutation, we use heuristics
                    output_dims[0] = inputs[0][0]  # Batch usually stays first
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            # Try to find matching dynamic dimension
                            found = False
                            for j in range(1, len(input_tensor_shape_list[0])):
                                if input_tensor_shape_list[0][j] == -1 and j != i:
                                    output_dims[i] = inputs[0][j]
                                    found = True
                                    break
                            if not found:
                                output_dims[i] = inputs[0][i] if i < len(inputs[0]) else expr_builder.constant(1)
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case20
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(output_tensor_shape_list[0]) >= 2 and \
                all(dim == -1 for dim in input_tensor_shape_list[0]) and all(dim == -1 for dim in output_tensor_shape_list[0]):
                # Case 21: Fully dynamic reshaping - all dimensions are dynamic
                # Example: input [-1,-1,-1] -> output [-1,-1,-1,-1]
                def compute_output_shapes_case21(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # Try to preserve some dimensions from input
                    for i in range(len(output_tensor_shape_list[0])):
                        if i < len(inputs[0]):
                            output_dims[i] = inputs[0][i]
                        else:
                            # For additional dimensions, use constant 1 as placeholder
                            output_dims[i] = expr_builder.constant(1)
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case21
            elif len(input_tensor_shape_list) == 2 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 2 and len(input_tensor_shape_list[1]) >= 2 and \
                input_tensor_shape_list[0][0] == -1 and input_tensor_shape_list[1][0] == -1 and \
                output_tensor_shape_list[0][0] == -1:
                # Case 22: Einsum - flexible tensor contraction
                # Example: input1 [-1,M,K], input2 [-1,K,N] -> output [-1,M,N]
                def compute_output_shapes_case22(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            # Try to find dimension from inputs
                            if i < len(inputs[0]):
                                output_dims[i] = inputs[0][i]
                            elif len(inputs) > 1 and i < len(inputs[1]):
                                output_dims[i] = inputs[1][i]
                            else:
                                output_dims[i] = expr_builder.constant(1)
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case22
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 1 and len(output_tensor_shape_list[0]) >= 1 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                len(output_tensor_shape_list[0]) < len(input_tensor_shape_list[0]):
                # Case 23: Flatten - collapse multiple dimensions into one
                # Example: input [-1,C,H,W] -> output [-1,C*H*W]
                def compute_output_shapes_case23(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    output_dims[0] = inputs[0][0]  # Batch
                    for i in range(1, len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1 and len(inputs[0]) > i:
                            # For flattened dimension, might be product of multiple input dims
                            output_dims[i] = inputs[0][i]
                            # Could multiply remaining dims, but we keep it simple
                            for j in range(i + 1, len(inputs[0])):
                                if input_tensor_shape_list[0][j] != -1:
                                    output_dims[i] = expr_builder.operation(trt.DimensionOperation.PROD, *output_dims[i], *expr_builder.constant(input_tensor_shape_list[0][j]))
                                else:
                                    output_dims[i] = expr_builder.operation(trt.DimensionOperation.PROD, *output_dims[i], *inputs[0][j])
                            break
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case23
            elif len(input_tensor_shape_list) == 2 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[1]) == 1 and input_tensor_shape_list[0][0] == -1 and \
                output_tensor_shape_list[0][0] == -1:
                # Case 24: OneHot encoding - data and depth
                # Example: indices [-1], depth [1] -> output [-1,num_classes]
                def compute_output_shapes_case24(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    # Copy input dimensions
                    for i in range(min(len(inputs[0]), len(output_tensor_shape_list[0]))):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    # Additional dimensions from depth or static
                    for i in range(len(inputs[0]), len(output_tensor_shape_list[0])):
                        output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case24
            elif len(input_tensor_shape_list) == 1 and len(output_tensor_shape_list) == 1 and \
                len(input_tensor_shape_list[0]) >= 4 and len(output_tensor_shape_list[0]) >= 4 and \
                input_tensor_shape_list[0][0] == -1 and output_tensor_shape_list[0][0] == -1 and \
                input_tensor_shape_list[0][1] != -1 and output_tensor_shape_list[0][1] != -1:
                # Case 25: Normalization (BatchNorm, LayerNorm, etc.) - preserves shape
                # Example: input [-1,C,H,W] -> output [-1,C,H,W]
                def compute_output_shapes_case25(inputs, expr_builder):
                    output_dims = trt.DimsExprs(len(output_tensor_shape_list[0]))
                    for i in range(len(output_tensor_shape_list[0])):
                        if output_tensor_shape_list[0][i] == -1:
                            output_dims[i] = inputs[0][i]
                        else:
                            output_dims[i] = expr_builder.constant(output_tensor_shape_list[0][i])
                    return [output_dims]

                compute_output_shapes = compute_output_shapes_case25
            else:
                compute_output_shapes = None

            return compute_output_shapes

        class DummyPlugin(DummyBasePlugin):

            def __init__(self):
                super().__init__()
                self.plugin_name = random_plugin_name
                self.plugin_version = "1"
                self.plugin_namespace = ""
                self.num_outputs = len(output_tensor_data_type_list)
                return

            def clone(self) -> trt.IPluginV3:
                cloned_plugin = DummyPlugin()
                cloned_plugin.__dict__.update(self.__dict__)
                return cloned_plugin

            def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
                return output_tensor_data_type_list

            def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
                return _compute_output_shapes(input_tensor_shape_list, output_tensor_shape_list, inputs, shape_inputs, expr_builder)

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
