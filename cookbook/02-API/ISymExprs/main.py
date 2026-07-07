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
from typing import List

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperDDS, case_mark, check_api_coverage, check_array, print_enumerated_members

@case_mark
def case_symexpr_classes():
    # trt.ISymExpr related
    sym_expr = trt.ISymExpr()
    print(f"{sym_expr = }")
    # check_api_coverage(trt.ISymExpr())  # Sanity check, unnecessary in normal workflow
    # sym_expr.dtype()
    # sym_expr.expr()
    # sym_expr.type()

    # trt.ISymExprs related
    # trt.ISymExprs has no constructor
    check_api_coverage(obj_class=trt.ISymExprs)
    print(f"{trt.ISymExprs.nbSymExprs = }")

    # trt.IDimensionExpr related
    # trt.IDimensionExpr has no constructor
    check_api_coverage(obj_class=trt.IDimensionExpr)  # Sanity check, unnecessary in normal workflow
    # trt.IDimensionExpr.get_constant_value()
    # trt.IDimensionExpr.is_constant()
    # trt.IDimensionExpr.is_size_tensor()

    # trt.DimsExprs related
    dim_exps = trt.DimsExprs(2)
    check_api_coverage(obj_class=trt.DimsExprs)  # Sanity check, unnecessary in normal workflow

    print(f"{dim_exps[0] = }, {dim_exps[1] = }")

    print_enumerated_members(trt.DimensionOperation)
    # | DimensionOperation |            Meaning            |
    # | :----------------: | :---------------------------: |
    # |        SUM         |          first + second       |
    # |        PROD        |          first * second       |
    # |        MAX         |     max(first, second)        |
    # |        MIN         |     min(first, second)        |
    # |        SUB         |          first - second       |
    # |       EQUAL        |     1 if first == second else 0 |
    # |        LESS        |     1 if first <  second else 0 |
    # |     FLOOR_DIV      |     floor(first / second)     |
    # |      CEIL_DIV      |     ceil(first / second)      |

class DimensionOperationPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    # A Data-Dependent-Shape (DDS) plugin
    # Input : inputT0 [batch, max_seq]      float32)
    # Output: outputT0 [batch, size_tensor] float32, second dim is data-dependent)
    #         outputT1 []                   (int32 scalar size tensor, value == max_seq here)

    def __init__(self):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.plugin_name = "DimensionOperation"
        self.plugin_version = "1"
        self.num_outputs = 2
        self.plugin_namespace = ""
        self._keep = []  # Keep device tensors alive until the CUDA stream is synchronized
        return

    def get_capability_interface(self, plugin_capability_type: trt.PluginCapabilityType) -> trt.IPluginCapability:
        return self

    def clone(self) -> trt.IPluginV3:
        cloned_plugin = DimensionOperationPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def configure_plugin(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> None:
        return

    def get_output_data_types(self, input_types: List[trt.DataType]) -> List[trt.DataType]:
        return [input_types[0], trt.int32]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        batch = inputs[0][0]
        max_seq = inputs[0][1]
        two = expr_builder.constant(2)

        # Exercise every member of trt.DimensionOperation through expr_builder.operation(...).
        # The results are throwaway expressions here, only used to demonstrate the whole enum.
        _ = expr_builder.operation(trt.DimensionOperation.SUM, max_seq, two)  # max_seq + 2
        _ = expr_builder.operation(trt.DimensionOperation.PROD, max_seq, two)  # max_seq * 2
        _ = expr_builder.operation(trt.DimensionOperation.MAX, max_seq, two)  # max(max_seq, 2)
        _ = expr_builder.operation(trt.DimensionOperation.MIN, max_seq, two)  # min(max_seq, 2)
        _ = expr_builder.operation(trt.DimensionOperation.SUB, max_seq, two)  # max_seq - 2
        _ = expr_builder.operation(trt.DimensionOperation.EQUAL, max_seq, two)  # max_seq == 2
        _ = expr_builder.operation(trt.DimensionOperation.LESS, max_seq, two)  # max_seq < 2
        opt_value = expr_builder.operation(trt.DimensionOperation.FLOOR_DIV, max_seq, two)  # floor(max_seq / 2)
        _ = expr_builder.operation(trt.DimensionOperation.CEIL_DIV, max_seq, two)  # ceil(max_seq / 2)

        # Declare the scalar output (index 1) as a size tensor so its runtime value can drive
        # the data-dependent second dimension of output 0. `opt_value` is the value used during
        # auto-tuning, `max_seq` is the upper bound.
        output0 = trt.DimsExprs(2)
        output0[0] = batch
        output0[1] = expr_builder.declare_size_tensor(1, opt_value, max_seq)

        # The size tensor itself is a rank-0 (scalar) tensor.
        output1 = trt.DimsExprs(0)
        return [output0, output1]

    def supports_format_combination(self, pos: int, in_out: List[trt.DynamicPluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos].desc
        if pos == 0:  # inputT0
            return desc.type == trt.float32 and desc.format == trt.TensorFormat.LINEAR
        elif pos == 1:  # outputT0
            return desc.type == in_out[0].desc.type and desc.format == trt.TensorFormat.LINEAR
        elif pos == 2:  # outputT1 (size tensor)
            return desc.type == trt.int32 and desc.format == trt.TensorFormat.LINEAR
        return False

    def get_workspace_size(self, dptd_in: List[trt.DynamicPluginTensorDesc], dptd_out: List[trt.DynamicPluginTensorDesc]) -> int:
        return 0

    def get_valid_tactics(self) -> List[int]:
        return [1]

    def set_tactic(self: trt.IPluginV3, tactic: int) -> None:
        return None

    def on_shape_change(self, ptd_in: List[trt.PluginTensorDesc], ptd_out: List[trt.PluginTensorDesc]) -> None:
        return None

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        batch, max_seq = input_desc[0].dims
        n_element = int(batch) * int(max_seq)
        buffer_size = n_element * np.dtype(np.float32).itemsize

        # output 0: copy the whole input (the data-dependent second dimension equals max_seq here)
        cudart.cudaMemcpyAsync(outputs[0], inputs[0], buffer_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)

        # output 1: write the scalar size tensor value (== max_seq)
        size_tensor = torch.tensor([int(max_seq)], dtype=torch.int32, device="cuda")
        self._keep.append(size_tensor)
        cudart.cudaMemcpyAsync(outputs[1], size_tensor.data_ptr(), np.dtype(np.int32).itemsize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream)
        return

    def attach_to_context(self, resource_context: trt.IPluginResourceContext) -> trt.IPluginV3:
        return self.clone()

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection([])

class DimensionOperationPluginCreator(trt.IPluginCreatorV3One):

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "DimensionOperation"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection, phase: trt.TensorRTPhase):
        return DimensionOperationPlugin()

@case_mark
def case_dimension_operation():
    # Build and run a Data-Dependent-Shape plugin whose `get_output_shapes` exercises every
    # member of `trt.DimensionOperation` and `trt.IExprBuilder.declare_size_tensor`.
    shape = [2, 4]
    data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
    trt_file = Path("model.trt")

    tw = TRTWrapperDDS(trt_file=trt_file)
    if tw.engine_bytes is None:  # Create engine from scratch
        plugin_creator = trt.get_plugin_registry().get_creator("DimensionOperation", "1", "")
        plugin = plugin_creator.create_plugin("DimensionOperation", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD)

        input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1])
        tw.profile.set_shape(input_tensor.name, [1, 1], shape, shape)

        layer = tw.network.add_plugin_v3([input_tensor], [], plugin)
        layer.name = "DimensionOperationPluginLayer"
        tensor0 = layer.get_output(0)
        tensor0.name = "outputT0"
        tensor1 = layer.get_output(1)
        tensor1.name = "outputT1"

        tw.build([tensor0, tensor1])
        tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()

    # outputT0 == inputT0 (copy), outputT1 == max_seq (the data-dependent size)
    check_array(tw.buffer["outputT0"][0], data["inputT0"], True, "outputT0")
    check_array(tw.buffer["outputT1"][0], np.array(shape[1], dtype=np.int32), True, "outputT1")

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    # We should only operate the resources once cross sessions
    plugin_registry = trt.get_plugin_registry()
    my_plugin_creator = DimensionOperationPluginCreator()
    if my_plugin_creator.name not in [creator.name for creator in plugin_registry.all_creators]:
        plugin_registry.register_creator(my_plugin_creator, "")

    # List APIs related to symbolic-expression classes
    case_symexpr_classes()
    # Exercise every trt.DimensionOperation member and declare_size_tensor via an IExprBuilder
    case_dimension_operation()

    print("Finish")
