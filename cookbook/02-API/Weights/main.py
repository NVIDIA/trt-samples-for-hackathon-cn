# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, grep_used_members

weight = trt.Weights(np.ones((1, 1, 1, 1), dtype=np.float32))

public_member = APIExcludeSet.analyze_public_members(weight, b_print=True)
grep_used_members(Path(__file__), public_member)

print(f"{weight.numpy() = }")
print(f"{weight.dtype = }")
print(f"{weight.nbytes = }")
print(f"{weight.size = }")

print("Finish")
