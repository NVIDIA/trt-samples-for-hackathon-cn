# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
