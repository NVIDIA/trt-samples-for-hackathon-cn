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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import MyDebugListener, TRTWrapperV1

data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}
expect_result = {"a_cute_tensor": data["inputT0"] * 3}  # the actual expected result is data*2, we set a wrong value here

tw = TRTWrapperV1()

tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
layer1 = tw.network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.SUM)
tensor1 = layer1.get_output(0)
tensor1.name = "a_cute_tensor"
layer2 = tw.network.add_elementwise(tensor1, tensor1, trt.ElementWiseOperation.SUM)

tw.network.mark_debug(tensor1)  # mark a tensor as debug-able
#tw.network.unmark_debug(tensor1)  # unmark a tensor as debug-able
print(f"{tw.network.is_debug_tensor(tensor1) = }")  # ensure one tensor is marked as debug-able

tw.build([layer2.get_output(0)])

tw.setup(data)

my_debug_listener = MyDebugListener(expect_result)
tw.context.set_debug_listener(my_debug_listener)  # set a debug listener for context
#debug_listener = context.get_debug_listener(MyDebugListener(expect_result))  # get a debug listener from context

tw.context.set_tensor_debug_state("a_cute_tensor", True)  # enable one debug tensor
#context.set_all_tensors_debug_state(True)  # enable all debug tensor

print(f"{tw.engine.is_debug_tensor('a_cute_tensor') = }")  # ensure one tensor is debug-able
print(f"{tw.context.get_debug_state('a_cute_tensor') = }")  # ensure one debug tensor is enabled

tw.infer()

# TODO: add usage of `unfused_tensors_debug_state`
