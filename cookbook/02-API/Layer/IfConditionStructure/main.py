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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, datatype_cast

@case_mark
def case_simple():
    """
    # Behave like:
    if tensor.reshape(-1)[0] != 0:
        return tensor * 2
    else:
        return tensor
    """
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}
    data1 = {"tensor": data["tensor"] - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    # Extract the scalar first element
    layer1 = tw.network.add_shuffle(tensor)
    layer1.reshape_dims = [-1]  # [Optional]
    layer2 = tw.network.add_slice(layer1.get_output(0), [0], [1], [1])
    layer2.start = [0]  # Reset later
    layer2.shape = [1]  # Reset later
    layer2.stride = [1]  # Reset later
    layer3 = tw.network.add_shuffle(layer2.get_output(0))
    layer3.reshape_dims = []  # [Optional]
    layer4 = tw.network.add_cast(layer3.get_output(0), trt.bool)

    if_structure = tw.network.add_if_conditional()
    # Input: condition tensor (rank 0, data type trt.DataType.BOOL); input tensors passed via add_input()
    # Outputs: output tensors from add_output(), one per add_output() call; outputs from ConditionLayer/IfConditionalInputLayer/IfConditionalOutputLayer.get_output() are None
    # Data type: condition tensor must be trt.DataType.BOOL; input/output tensor data types are unrestricted
    # Shape: condition tensor must have rank 0 (scalar); input and output tensors may have any shape
    if_structure.name = "A cute If Condition Structure"  # [Optional] Default: auto-generated name

    # Use the tensor created by the `add_input` API in the condition body to avoid it computing twice in both true and false branches
    layer_input = if_structure.add_input(tensor)
    if_structure.set_condition(layer4.get_output(0))
    # Branch of condition is true
    layer_true = tw.network.add_elementwise(layer_input.get_output(0), layer_input.get_output(0), trt.ElementWiseOperation.SUM)
    # Branch of condition is false
    layer_false = tw.network.add_identity(layer_input.get_output(0))
    layer_output = if_structure.add_output(layer_true.get_output(0), layer_false.get_output(0))

    check_api_coverage(if_structure)  # Sanity check, unnecessary in normal workflow

    tw.build([layer_output.get_output(0)])
    tw.setup(data)
    tw.infer()

    tw.setup(data1)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using if condition structure
    case_simple()

    print("Finish")
