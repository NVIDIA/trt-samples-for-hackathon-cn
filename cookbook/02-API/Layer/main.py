#
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

import sys

import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1

shape = [3, 4, 5]

tw = TRTWrapperV1()
tw.config.set_flag(trt.BuilderFlag.INT8)  # use Int8 mode in this example for using certain APIs

tensor = tw.network.add_input("tensor", trt.float32, [-1] * len(shape))
tw.profile.set_shape(tensor.name, [1, 1, 1], shape, shape)
tw.config.add_optimization_profile(tw.profile)

# Add a layer
layer = tw.network.add_identity(tensor)
layer.name = "Identity Layer"
layer.metadata = "My message"
layer.precision = trt.int8
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.reset_output_type(0)
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.get_output(0).dynamic_range = [-128, 128]

# Just for ensuring the network is self-consistent
tw.build([layer.get_output(0)])

# Print information of the layer
print(f"{layer.name = }")
print(f"{layer.__class__ = }")  # For type casting from ILayer to exact type of layer
print(f"{layer.metadata = }")
print(f"{layer.type = }")
print(f"{layer.precision = }")
print(f"{layer.precision_is_set = }")
print(f"{layer.num_inputs = }")
for i in range(layer.num_inputs):
    print(f"    layer.get_input({i}) = {layer.get_input(i)}")  # get input tensor from layer
print(f"{layer.num_outputs = }")
for i in range(layer.num_outputs):
    print(f"    layer.get_output({i}) = {layer.get_output(i)}")  # get output tensor from layer
    print(f"    layer.get_output_type({i}) = {layer.get_output_type(i)}")
    print(f"    layer.output_type_is_set({i}) = {layer.output_type_is_set(i)}")

layer.set_input(0, tensor)  # set input tensor rather than add_* API
layer.reset_precision()
print(f"{layer.precision = }")
print(f"{layer.precision_is_set = }")

print("Finish")
