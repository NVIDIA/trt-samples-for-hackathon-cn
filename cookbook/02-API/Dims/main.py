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

import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, grep_used_members

dim = trt.Dims([1, 3, 4, 5])  # input argument is `collections.abc.Sequence[typing.SupportsInt]`

public_member = APIExcludeSet.analyze_public_members(dim, b_print=True)
grep_used_members(Path(__file__), public_member)

print(f"{dim = }")
print(f"{dim.MAX_DIMS = }")
print(f"{list(dim) = }")
print(f"{tuple(dim) = }")

print(f"{trt.Dims() = }")
print(f"{trt.Dims2(28, 28) = }")  # input argument is `typing.SupportsInt`, no Sequence
print(f"{trt.Dims3(28, 28, 28) = }")
print(f"{trt.Dims4(28, 28, 28, 28) = }")

dim_hw = trt.DimsHW(28, 28)
print(f"{dim_hw = }, {dim_hw.h = }, {dim_hw.w = }")

print("Finish")
