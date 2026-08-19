# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""The Triton kernels that get compiled ahead of time into the two plugins.

Nothing here is TensorRT-aware: these are ordinary `@triton.jit` kernels, and `triton.tools.compile`
is run on this file from the Makefile / `plugin_gen.py`. This file is never imported at run time --
the cubin is baked into the `.so`.
"""

import triton
import triton.language as tl

@triton.jit
def add_scalar_kernel(x_ptr, y_ptr, scalar, n_element, BLOCK_SIZE: tl.constexpr):
    """`y = x + scalar`, the cookbook's standard plugin operator.

    `scalar` is a **runtime fp32 argument**, which is what walks into the ABI bug documented in
    `README.md`: the generated C launcher declares it `double`.
    """
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_element
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(y_ptr + offset, x + scalar, mask=mask)

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_element, BLOCK_SIZE: tl.constexpr):
    """Exact GELU, `y = x * 0.5 * (1 + erf(x / sqrt(2)))`.

    Used by `plugin_gen.py` to prove the generator is not hard-wired to `add_scalar`: different
    operator, different argument list, no attribute at all, and enough arithmetic that the
    `BLOCK_SIZE` / `num_warps` choice actually shows up in the tactic timings.
    """
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_element
    x = tl.load(x_ptr + offset, mask=mask)
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(y_ptr + offset, y, mask=mask)
