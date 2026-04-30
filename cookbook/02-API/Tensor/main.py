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
from tensorrt_cookbook import (TRTWrapperV1, check_api_coverage, format_to_string)

shape = [2, 3, 4, 5]
data = (np.arange(np.prod(shape), dtype=np.float32) / np.prod(shape) * 128).reshape(shape)

tw = TRTWrapperV1()

tw.builder_config.set_flag(trt.BuilderFlag.INT8)

inputT0 = tw.network.add_input("inputT0", trt.float32, [-1 for _ in shape])
inputT0.set_dimension_name(0, "Batch Size")
tw.profile.set_shape(inputT0.name, [1] + shape[1:], shape, [4] + shape[1:])

layer = tw.network.add_identity(inputT0)

tensor = layer.get_output(0)
tensor.name = "Identity Layer Output Tensor 0"
tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

check_api_coverage(tensor)  # Sanity check, unnecessary in normal workflow

print(f"\n{'=' * 64} Usage show")

print(f"{tensor.name = }")
print(f"{tensor.shape = }")
print(f"{tensor.location = }")
print(f"{tensor.dtype = }")
print(f"{tensor.broadcast_across_batch = }")  # deprecated
print(f"tensor.allowed_formats = {format_to_string(tensor.allowed_formats)}")

print(f"{tensor.is_execution_tensor = }")
print(f"{tensor.is_shape_tensor = }")
print(f"{tensor.is_network_input = }")
print(f"{tensor.is_network_output = }")
print(f"{inputT0.get_dimension_name(0) = }")  # Only for input tensor

tensor.set_dynamic_range(0, 1)  # deprecated
print(f"{tensor.dynamic_range = }")  # deprecated
tensor.reset_dynamic_range()  # deprecated

print("Finish")
