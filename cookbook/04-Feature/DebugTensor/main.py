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
from tensorrt_cookbook import CookbookDebugListener, TRTWrapperV1

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

my_debug_listener = CookbookDebugListener(expect_result)
tw.context.set_debug_listener(my_debug_listener)  # set a debug listener for context
#debug_listener = context.get_debug_listener(CookbookDebugListener(expect_result))  # get a debug listener from context

tw.context.set_tensor_debug_state("a_cute_tensor", True)  # enable one debug tensor
#context.set_all_tensors_debug_state(True)  # enable all debug tensor

print(f"{tw.engine.is_debug_tensor('a_cute_tensor') = }")  # ensure one tensor is debug-able
print(f"{tw.context.get_debug_state('a_cute_tensor') = }")  # ensure one debug tensor is enabled

tw.infer()

# ------------------------------------------------------------------------------
# Usage of `mark_unfused_tensors_as_debug_tensors` + `unfused_tensors_debug_state`
# ------------------------------------------------------------------------------
# `mark_debug` prevents the marked tensor from being fused, which can change the
# optimized graph and hurt performance. `mark_unfused_tensors_as_debug_tensors`
# instead marks every tensor that survives fusion as a debug tensor, so it does
# not disturb fusion decisions (performance is preserved). The trade-off: these
# tensors are reported to the DebugListener by their internal (post-fusion) names
# rather than the names used in the NetworkDefinition, and they cannot be queried
# with `is_debug_tensor`.
tw2 = TRTWrapperV1()

tensor = tw2.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
layer1 = tw2.network.add_elementwise(tensor, tensor, trt.ElementWiseOperation.SUM)
layer2 = tw2.network.add_elementwise(layer1.get_output(0), layer1.get_output(0), trt.ElementWiseOperation.SUM)

print(f"{tw2.network.mark_unfused_tensors_as_debug_tensors() = }")  # mark all unfused tensors as debug tensors
#tw2.network.unmark_unfused_tensors_as_debug_tensors()  # reverse operation

tw2.build([layer2.get_output(0)])
tw2.setup(data)

tw2.context.set_debug_listener(CookbookDebugListener())  # internal names are unknown here, so we just print them

tw2.context.unfused_tensors_debug_state = True  # enable emitting of all unfused debug tensors at runtime
print(f"{tw2.context.unfused_tensors_debug_state = }")  # read back the state

tw2.infer()
