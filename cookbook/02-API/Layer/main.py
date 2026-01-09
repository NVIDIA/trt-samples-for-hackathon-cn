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

import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, TRTWrapperV1, layer_dynamic_cast

shape = [3, 4, 5]

tw = TRTWrapperV1()

tensor = tw.network.add_input("tensor", trt.float32, [-1] * len(shape))
tw.profile.set_shape(tensor.name, [1, 1, 1], shape, shape)
tw.config.add_optimization_profile(tw.profile)

# Add a layer
layer = tw.network.add_identity(tensor)

callback_member, callable_member, attribution_member = APIExcludeSet.split_members(layer)
print(f"\n{'=' * 64} Members of trt.ILayer:")
print(f"{len(callback_member):2d} Members to get/set common/callback classes: {callback_member}")
print(f"{len(callable_member):2d} Callable methods: {callable_member}")
print(f"{len(attribution_member):2d} Non-callable attributions: {attribution_member}")

layer.name = "Identity Layer"
layer.metadata = "My message"

layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.set_input(0, tensor)  # Add input tensors for some kind of layers

# Print information of the layer
print(f"{layer.name = }")
print(f"{layer.__class__ = }")  # For type casting from ILayer to exact type of layer
print(f"{layer.metadata = }")
print(f"{layer.type = }")

print(f"{layer.precision = }")
print(f"{layer.precision_is_set = }")  # deprecated
layer.reset_precision()  # deprecated

print(f"{layer.num_inputs = }")
for i in range(layer.num_inputs):
    print(f"    layer.get_input({i}) = {layer.get_input(i)}")  # Get input tensor from layer
print(f"{layer.num_outputs = }")
for i in range(layer.num_outputs):
    print(f"    layer.get_output({i}) = {layer.get_output(i)}")  # Get output tensor from layer
    print(f"    layer.get_output_type({i}) = {layer.get_output_type(i)}")
    layer.set_output_type(0, trt.float32)  # deprecated
    print(f"    layer.output_type_is_set({i}) = {layer.output_type_is_set(i)}")  # deprecated
    layer.reset_output_type(i)  # deprecated

# Dynamic cast from ILayer to exact type of layer
layer = tw.network.get_layer(0)
print(f"{type(layer) = }")
layer_dynamic_cast(layer)
print(f"{type(layer) = }")

layer_list = []
for name, (value, desc) in trt.LayerType.__entries.items():
    layer_list.append([int(value), name, desc])

for value_int, name, desc in sorted(layer_list, key=lambda x: x[0]):
    print(f"{value_int:2d}: {name:18s}, {desc}")

print("Finish")
