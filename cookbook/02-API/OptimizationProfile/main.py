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

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperShapeInput

shape = [3, 4, 5]
input_data = {}
input_data["inputT0"] = np.zeros(np.prod(shape), dtype=np.float32).reshape(shape)  # Execution input tensor
input_data["inputT1"] = np.array(shape, dtype=np.int32)  # Shape input tensor

tw = TRTWrapperShapeInput()

tensor0 = tw.network.add_input("inputT0", trt.float32, [-1 for _ in shape])
tensor1 = tw.network.add_input("inputT1", trt.int32, [len(shape)])
tw.profile.set_shape(tensor0.name, [1 for _ in shape], shape, shape)
tw.profile.set_shape_input(tensor1.name, [1 for _ in shape], shape, shape)

print(f"{tw.profile.get_shape(tensor0.name) = }")
print(f"{tw.profile.get_shape_input(tensor1.name) = }")

print("Finish")
