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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, grep_used_members

weight = trt.Weights(np.ones((1, 1, 1, 1), dtype=np.float32))

public_member = APIExcludeSet.analyze_public_members(weight)
grep_used_members(Path(__file__), public_member)

print(f"{weight.numpy() = }")
print(f"{weight.dtype = }")
print(f"{weight.nbytes = }")
print(f"{weight.size = }")

print("Finish")
